// sis/static/js/risk_badge.js
window.SIS_RISK = window.SIS_RISK || {};
window.SIS_RISK.fetchBadge = async function(btn){
  const slot = btn.parentElement.querySelector('[data-risk-slot]');
  const fd = new FormData();
  fd.append('goal', btn.getAttribute('data-risk-goal'));
  fd.append('reply', btn.getAttribute('data-risk-reply'));
  fd.append('model_alias', btn.getAttribute('data-risk-model') || 'chat-hrm');
  fd.append('monitor_alias', btn.getAttribute('data-risk-monitor') || 'tiny-monitor');
  try {
    const res = await fetch('/risk/api/analyze', { method: 'POST', body: fd });
    if(!res.ok){ throw new Error('HTTP '+res.status); }
    const rec = await res.json();
    slot.innerHTML = `
      <img src="${rec.badge_svg}" width="64" height="64" style="border-radius:12px;border:1px solid #ddd" alt="risk badge" />
      <span class="badge ${rec.decision==='RISK' ? 'bg-danger' : (rec.decision==='WATCH' ? 'bg-warning text-dark' : 'bg-success')}">${rec.decision}</span>
    `;
  } catch(err){
    slot.textContent = '⚠︎ Risk check failed';
    console.error('Risk badge fetch error', err);
  }
};
