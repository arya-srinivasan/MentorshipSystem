let fullTranscript = '';
let segments = [];
let isRunning = false;

document.getElementById('startBtn').addEventListener('click', async () => {
    if (isRunning) return;
    isRunning = true;
    fullTranscript = '';
    segments = [];
    document.getElementById('output').textContent = '';
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.runtime.sendMessage({ action: 'start', tabId: tab.id });
    document.getElementById('status').textContent = 'Transcribing...';
  });
  
  document.getElementById('stopBtn').addEventListener('click', () => {
    isRunning = false;
    chrome.runtime.sendMessage({ action: 'stop' });
    document.getElementById('status').textContent = 'Stopped';
  });

document.getElementById('downloadBtn').addEventListener('click', () => {
    if (!fullTranscript) return;
    const blob = new Blob([JSON.stringify({ 
      created: new Date().toISOString(),
      segments: segments
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${new Date().toISOString().slice(0,19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'transcript') {
      if (msg.segments) segments = msg.segments;
      if (msg.isFinal) {
        fullTranscript = segments.map(s => s.text).join(' ');
      }
      const output = document.getElementById('output');
      output.textContent = msg.isFinal ? fullTranscript : fullTranscript + ' ' + msg.text;
      output.scrollTop = output.scrollHeight;
    }
  });