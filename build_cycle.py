"""
build_cycles.py  (v73 조건 반영)
FFCell raw 센서 로그 → 사이클 단위 분석표 + 센서 패턴 + 성능지표 생성

산출물:
  cycles.csv            사이클 1행 (라벨/피처/이상스코어/규칙기반예측)
  patterns.csv          로봇×클래스 평균 센서 파형 (정상 vs 결함 비교용)
  feature_importance.csv 피처 중요도
  metrics.json          정확도/탐지균형/결함탐지/클래스별 recall/혼동행렬

핵심 처리:
  - 세션 = CycleCount 감소(reset) 경계로 분리, id = "{session}_{count}"
  - v73 제외 10건 적용 (327 → 317)
  - R02 그리퍼 계열은 오염(라벨누수) 피처라 모델/이상탐지에서 제외 (부분일치)
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

ROBOTS = ["R01", "R02", "R03", "R04"]
CONTAMINATED = ["R02", "CycleCount"]          # 부분일치로 제외
EXCLUDE_IDS = ["2_0", "4_0",                  # ① 지속시간 QA (불완전)
               "1_1", "1_2",                  # ② 센서 QA (워밍업 결측)
               "0_1", "0_2", "0_3",           # ③ 세션0 (커미셔닝 정책 제외)
               "5_182",                       # ④ dropna (피처 NaN)
               "5_253", "5_265"]              # ⑤ 라벨오류 (확정 제외)


def primary_class(desc_series):
    v = desc_series.dropna().astype(str)
    v = v[v.str.strip() != ""]
    if len(v) == 0:
        return "Normal"
    # 원본 Description 문자열을 그대로 라벨로 사용 (4종: Normal / NoNose /
    # NoNose,NoBody2 / NoNose,NoBody2,NoBody1). 공백만 정리.
    return ",".join(p.strip() for p in v.value_counts().index[0].split(","))


def load_raw(root):
    cyc = pd.read_csv(os.path.join(root, "FFCell_CycleManagement.csv"), low_memory=False)
    df = pd.DataFrame({"_time": pd.to_datetime(cyc["_time"], errors="coerce")})
    cc = pd.to_numeric(cyc["Q_Cell_CycleCount"], errors="coerce")
    df["cc"] = cc
    df["session"] = (cc.diff() < 0).cumsum().fillna(0).astype(int)
    df["Description"] = cyc["Description"]
    for r in ROBOTS:
        d = pd.read_csv(os.path.join(root, f"{r}_Data.csv"),
                        usecols=[f"I_{r}_Gripper_Load"], low_memory=False)
        df[r] = pd.to_numeric(d[f"I_{r}_Gripper_Load"], errors="coerce")
    saf = pd.read_csv(os.path.join(root, "FFCell_SafetyManagement.csv"),
                      usecols=["CabinetESTOP", "I_HMI_EStop_Status"], low_memory=False)
    df["estop"] = (saf["CabinetESTOP"].astype(str).str.upper() == "TRUE") | \
                  (saf["I_HMI_EStop_Status"].astype(str).str.upper() == "FALSE")
    df["id"] = df["session"].astype(str) + "_" + df["cc"].astype("Int64").astype(str)
    return df


def build(root):
    df = load_raw(root)
    rows, patt = [], []
    for cid, g in df.groupby("id"):
        label = primary_class(g["Description"])
        rec = {"id": cid, "session": int(g["session"].iloc[0]),
               "label": label, "defect": label != "Normal",
               "n_samples": len(g),
               "duration_s": (g["_time"].max() - g["_time"].min()).total_seconds(),
               "n_estop": int(g["estop"].sum())}
        for r in ROBOTS:
            v = g[r].dropna()
            rec[f"{r}_mean"] = v.mean(); rec[f"{r}_max"] = v.max()
            rec[f"{r}_std"] = v.std()
            rec[f"{r}_rms"] = float(np.sqrt(np.mean(v**2))) if len(v) else np.nan
        rows.append(rec)
    cyc = pd.DataFrame(rows)

    # --- v73 제외 (327 → 317) ---
    before = len(cyc)
    cyc = cyc[~cyc["id"].isin(EXCLUDE_IDS)].reset_index(drop=True)
    print(f"제외 적용: {before} → {len(cyc)} (−{before-len(cyc)})")

    # --- 모델/이상탐지 피처 (오염 피처 부분일치 제외) ---
    all_feats = [f"{r}_{s}" for r in ROBOTS for s in ("mean", "max", "std", "rms")] + ["n_estop"]
    feats = [f for f in all_feats if not any(c in f for c in CONTAMINATED)]
    X = cyc[feats].apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.fillna(X.median())
    y = np.asarray(cyc["label"].values, dtype=object)

    # --- 규칙기반(얕은 결정트리) 분류: 교차검증으로 정직하게 평가 ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    pred = cross_val_predict(clf, X, y, cv=skf)
    cyc["pred"] = pred
    labels_order = sorted(cyc["label"].unique())
    metrics = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_recall": round(float(recall_score(y, pred, average="macro")), 4),
        "defect_detection": round(float(
            ((pred != "Normal") & (y != "Normal")).sum() / max((y != "Normal").sum(), 1)), 4),
        "per_class_recall": {l: round(float(recall_score(y == l, pred == l)), 4)
                             for l in labels_order},
        "labels": labels_order,
        "confusion": confusion_matrix(y, pred, labels=labels_order).tolist(),
        "n_cycles": int(len(cyc)), "n_features": len(feats),
    }

    # --- 피처 중요도 (RF) ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
    fi = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False)

    # --- Mahalanobis 이상 스코어 (정상 분포, R02 제외됨) ---
    normal = X[cyc["label"] == "Normal"]
    mu, inv = normal.mean().values, np.linalg.pinv(np.cov(normal.values, rowvar=False))
    diff = X.values - mu
    cyc["anomaly_score"] = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    thr = np.percentile(cyc.loc[cyc["label"] == "Normal", "anomaly_score"], 97.5)
    cyc["threshold"] = thr
    cyc["is_anomaly"] = cyc["anomaly_score"] > thr

    # --- 센서 패턴: 로봇×클래스 평균 파형 (사이클 내 상대시간 0~1, 100구간) ---
    keep = set(cyc["id"])
    for cid, g in df[df["id"].isin(keep)].groupby("id"):
        lab = cyc.loc[cyc["id"] == cid, "label"].iloc[0]
        t = g["_time"].values.astype("datetime64[ns]").astype(float)
        t = (t - t.min()) / max((t.max() - t.min()), 1)
        bins = np.clip((t * 100).astype(int), 0, 99)
        for r in ROBOTS:
            tmp = pd.DataFrame({"bin": bins, "v": g[r].values})
            m = tmp.groupby("bin")["v"].mean()
            for b, val in m.items():
                patt.append({"robot": r, "klass": lab, "t_bin": int(b), "value": float(val)})
    patterns = pd.DataFrame(patt).groupby(["robot", "klass", "t_bin"], as_index=False)["value"].mean()


    # --- 드릴다운용 파형 (클라우드 배포용, 사이클당 ~150포인트로 다운샘플) ---
    dd_rows = []
    for cid, g in df[df["id"].isin(keep)].groupby("id"):
        g = g.reset_index(drop=True)
        step = max(1, len(g) // 150)
        gs = g.iloc[::step]
        t = gs["_time"].values.astype("datetime64[ns]").astype("float64")
        t = (t - t.min()) / 1e9  # 초
        for i, (_, rr) in enumerate(gs.iterrows()):
            dd_rows.append({"id": cid, "t": round(float(t[i]), 2),
                            "R01": rr["R01"], "R02": rr["R02"],
                            "R03": rr["R03"], "R04": rr["R04"]})
    drill = pd.DataFrame(dd_rows)

    return cyc, fi, patterns, metrics, drill


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args()
    cyc, fi, patterns, metrics, drill = build(a.root)
    cyc.to_csv("cycles.csv", index=False)
    fi.to_csv("feature_importance.csv", index=False)
    patterns.to_csv("patterns.csv", index=False)
    drill.to_csv("drilldown.csv", index=False)
    json.dump(metrics, open("metrics.json", "w"), ensure_ascii=False, indent=2)
    print("saved cycles / feature_importance / patterns / metrics / drilldown")
    print("label dist:", cyc["label"].value_counts().to_dict())
    print("accuracy=%.3f macro_recall=%.3f defect_detect=%.3f"
          % (metrics["accuracy"], metrics["macro_recall"], metrics["defect_detection"]))
    print("top feature:", fi.iloc[0].to_dict())
