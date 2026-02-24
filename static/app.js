const $ = (id) => document.getElementById(id);

function setStatus(el, msg) {
  el.textContent = msg;
}

function fillField(id, value) {
  $(id).value = value || "";
}

function setConf(id, value, source) {
  if (!$(id)) return;
  if (value === null || value === undefined) {
    $(id).textContent = "";
    return;
  }
  const pct = Math.round((Number(value) || 0) );
  const src = "";
$(id).textContent = `Confidence: ${pct}%`;
}

$('btnExtract').addEventListener('click', async () => {
  const file = $('file').files[0];
  if (!file) {
    alert('Please upload a DBS PDF or image first.');
    return;
  }

  setStatus($('extractStatus'), 'Extracting...');

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/dbs/extract', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    fillField('certificate_number', data.certificate_number);
    fillField('surname_extracted', data.surname);

    fillField('dob_day', data.dob_day);
    fillField('dob_month', data.dob_month);
    fillField('dob_year', data.dob_year);
setConf('conf_cert', data.confidence?.certificate_number, data.source?.certificate_number);
    setConf('conf_surname', data.confidence?.surname, data.source?.surname);
    setConf('conf_dob', data.confidence?.dob, data.source?.dob);
setStatus($('extractStatus'), 'Done. Review/edit fields if needed.');
  } catch (e) {
    console.error(e);
    setStatus($('extractStatus'), 'Extraction failed. Check console/logs.');
    alert('Extraction failed. If this is a scanned PDF/image, set GEMINI_API_KEY on Render.');
  }
});

$('btnRun').addEventListener('click', async () => {
  const payload = {
    organisation_name: $('org').value.trim(),
    forename: $('forename').value.trim(),
    surname_user: $('surname_user').value.trim(),

    certificate_number: $('certificate_number').value.trim(),
    surname_extracted: $('surname_extracted').value.trim(),

    dob_day: $('dob_day').value.trim(),
    dob_month: $('dob_month').value.trim(),
    dob_year: $('dob_year').value.trim(),
  };

  if (!payload.organisation_name || !payload.forename || !payload.surname_user) {
    alert('Please fill Organisation Name, Forename, and Surname (manual inputs).');
    return;
  }
  if (!payload.certificate_number || !payload.surname_extracted || !payload.dob_day || !payload.dob_month || !payload.dob_year) {
    alert('Please extract and ensure Certificate Number, Surname (from DBS), and DOB are filled.');
    return;
  }

  setStatus($('runStatus'), 'Running... this may take a bit.');

  try {
    const res = await fetch('/dbs/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();

    // Get filename from Content-Disposition
    let filename = 'DBS-Check.pdf';
    const cd = res.headers.get('Content-Disposition') || res.headers.get('content-disposition');
    if (cd) {
      const match = cd.match(/filename="?([^";]+)"?/i);
      if (match) filename = match[1];
    }

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);

    setStatus($('runStatus'), 'Downloaded.');
  } catch (e) {
    console.error(e);
    setStatus($('runStatus'), 'Run failed. Check logs.');
    alert('Run failed. If DBS website changed or blocks automation, we will need to update selectors/flow.');
  }
});
