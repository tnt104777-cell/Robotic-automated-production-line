# FFCell 로켓 조립 품질 대시보드

로켓 부품 조립 라인 센서 데이터 기반 결함 분류 · 품질 모니터링 대시보드 (Streamlit).

## 구성
- **① 종합 현황** — 조립 카운트, 분류 성능, 피처중요도·recall·혼동행렬
- **② 센서·로봇 움직임** — 로봇별 평균 파형 (정상 vs 결함)
- **③ 사이클 드릴다운** — 사이클 선택 → 센서 파형 + 판정
- **④ 이미지 CV** — 멀티모달 확장 (개발 중)

## 로컬 실행
```
pip install -r requirements.txt
python build_cycle.py --root .   # raw CSV 필요 (한 번만)
streamlit run app.py
```

## 배포 (Streamlit Cloud)
raw 센서 CSV는 용량이 커서 올리지 않습니다. 대신 build_cycle.py가 생성한
결과물(cycles.csv, patterns.csv, feature_importance.csv, metrics.json, drilldown.csv)을
함께 올리면 app.py가 그것만 읽어 작동합니다.
