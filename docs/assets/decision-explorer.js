/* 가상 관측에 대한 분기만 계산합니다. 외부 서버나 모델은 호출하지 않습니다. */
(function (root) {
  'use strict';
  function classify(row, rubric, thresholds) {
    const checked = [...new Set([...Object.keys(row.record.target), 'continuity', 'visibility'])].sort();
    const reasons = [];
    const mismatches = [];
    for (const key of checked) {
      const answer = row.raw.answers[key];
      const values = Object.values(answer.probabilities).sort((a, b) => b - a);
      const probability = answer.probabilities[answer.choice];
      const margin = values[0] - values[1];
      if (answer.choice === 'unknown') reasons.push(key + ':판단_불가');
      if (probability < thresholds.min_probability || margin < thresholds.min_margin) {
        reasons.push(key + ':낮은_확신_또는_선택지_경합');
      }
      if (rubric.questions[key].temporal) {
        const gap = row.evidence.max_gap_s;
        const hz = row.record.source_observation_hz;
        if (!Number.isFinite(gap) || gap < 0 || gap > thresholds.max_gap_s) reasons.push(key + ':샘플링_간격_부족');
        if (hz == null || 1 / hz > thresholds.max_gap_s) reasons.push(key + ':원본_시간해상도_미확인_또는_부족');
      }
      if (Object.hasOwn(row.record.target, key) && answer.choice !== 'unknown' && answer.choice !== row.record.target[key]) {
        mismatches.push(key);
      }
    }
    const artifact = row.raw.answers.continuity.choice === 'artifact';
    const disposition = reasons.length ? 'review' : artifact ? 'quarantine' : mismatches.length ? 'relabel_candidate' : 'candidate';
    return {disposition, reasons, mismatches};
  }
  root.DecisionExplorer = {classify};
  if (typeof document === 'undefined') return;
  const app = document.getElementById('decision-explorer');
  if (!app) return;
  const data = JSON.parse(document.getElementById('fixture-data').textContent);
  const ids = ['min_probability', 'min_margin', 'max_gap_s'];
  const names = {candidate: '학습 후보', relabel_candidate: '재분류 후보', review: '판단 보류', quarantine: '결함 검토'};
  function render() {
    const thresholds = Object.fromEntries(ids.map(id => [id, Number(document.getElementById(id).value)]));
    for (const id of ids) document.getElementById(id + '-value').textContent = thresholds[id].toFixed(2) + (id === 'max_gap_s' ? '초' : '');
    const counts = {candidate: 0, relabel_candidate: 0, review: 0, quarantine: 0};
    const rows = data.rows.map(row => {
      const result = classify(row, data.rubric, thresholds);
      counts[result.disposition]++;
      return {row, result};
    });
    const chart = document.getElementById('counts-chart');
    chart.replaceChildren();
    const summary = Object.keys(counts).map(key => names[key] + ' ' + counts[key] + '개').join(' · ');
    chart.setAttribute('aria-label', summary);
    for (const key of Object.keys(counts)) {
      if (!counts[key]) continue;
      const segment = document.createElement('div');
      segment.className = 'segment ' + key;
      segment.style.flexGrow = counts[key];
      segment.textContent = String(counts[key]);
      segment.setAttribute('aria-hidden', 'true');
      chart.appendChild(segment);
    }
    document.getElementById('summary').textContent = summary;
    const body = document.getElementById('result-rows');
    body.replaceChildren();
    for (const {row, result} of rows) {
      const tr = document.createElement('tr');
      const reason = result.reasons.length
        ? [...new Set(result.reasons.map(r => r.includes('원본_') ? '원본 시간 근거 부족' : r.includes('샘플링_') ? '프레임 공백 초과' : r.includes('판단_불가') ? '관측 답변이 unknown' : '점수 기준 미달'))].join(', ')
        : result.disposition === 'quarantine' ? '명백한 생성 결함 관측'
        : result.mismatches.length ? '목표와 끼어들기 관측 불일치'
        : '목표와 관측 일치';
      [row.label, names[result.disposition], reason].forEach((value, index) => {
        const td = document.createElement('td');
        if (index === 1) {
          const marker = document.createElement('span');
          marker.className = 'swatch ' + result.disposition;
          marker.setAttribute('aria-hidden', 'true');
          td.appendChild(marker);
        }
        td.appendChild(document.createTextNode(value));
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
    document.getElementById('time-note').textContent = thresholds.max_gap_s >= 0.50
      ? '허용 공백을 늘려 2Hz 자료도 시간 조건을 만족할 수 있게 했습니다. 관측 정보가 늘어난 것은 아닙니다.'
      : '원본 2Hz의 관측 공백은 0.50초이므로 현재 시간 기준을 만족하지 못합니다.';
  }
  ids.forEach(id => document.getElementById(id).addEventListener('input', render));
  render();
})(typeof globalThis === 'undefined' ? this : globalThis);
