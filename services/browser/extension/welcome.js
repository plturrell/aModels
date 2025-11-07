let currentStep = 1;

function updateStepIndicator() {
  const dots = document.querySelectorAll('.step-dot');
  dots.forEach((dot, index) => {
    if (index + 1 === currentStep) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function showStep(stepNumber) {
  document.querySelectorAll('.step').forEach(step => {
    step.classList.remove('active');
  });
  
  const targetStep = document.getElementById(`step${stepNumber}`);
  if (targetStep) {
    targetStep.classList.add('active');
  }
  
  currentStep = stepNumber;
  updateStepIndicator();
}

function nextStep() {
  if (currentStep < 3) {
    showStep(currentStep + 1);
  }
}

function prevStep() {
  if (currentStep > 1) {
    showStep(currentStep - 1);
  }
}

async function testConnection() {
  const input = document.getElementById('gateway-url-input');
  const status = document.getElementById('connection-status');
  const testBtn = document.getElementById('test-btn');
  
  const gatewayUrl = input.value.trim() || 'http://localhost:8000';
  
  // Show loading state
  status.className = 'status loading visible';
  status.innerHTML = '<div class="spinner"></div><span>Testing connection...</span>';
  testBtn.disabled = true;
  
  try {
    // Test the connection
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${gatewayUrl}/healthz`, {
      signal: controller.signal,
      method: 'GET'
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      // Success! Save the URL and move to next step
      await chrome.storage.sync.set({ 
        gatewayBaseUrl: gatewayUrl,
        setupCompleted: true,
        setupDate: new Date().toISOString()
      });
      
      status.className = 'status success visible';
      status.innerHTML = '✅ Connection successful! Gateway is responding.';
      
      // Auto-advance after a brief moment
      setTimeout(() => {
        showStep(3);
      }, 1500);
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (error) {
    // Show error
    testBtn.disabled = false;
    status.className = 'status error visible';
    
    if (error.name === 'AbortError') {
      status.innerHTML = `⚠️ Connection timed out. Please check:<br>
        • Is the gateway running?<br>
        • Is the URL correct?<br>
        • Check firewall settings`;
    } else if (error.message === 'Failed to fetch') {
      status.innerHTML = `⚠️ Cannot reach gateway. Please check:<br>
        • Is the gateway running?<br>
        • Is the URL correct? (${gatewayUrl})<br>
        • Try: <code>curl ${gatewayUrl}/healthz</code>`;
    } else {
      status.innerHTML = `⚠️ Connection failed: ${error.message}<br>
        Check the gateway logs and try again.`;
    }
  }
}

async function finish() {
  // Mark setup as completed
  await chrome.storage.sync.set({ 
    setupCompleted: true,
    setupDate: new Date().toISOString()
  });
  
  // Close this tab and open the popup
  window.close();
}

// Allow Enter key to advance
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    if (currentStep === 1) {
      nextStep();
    } else if (currentStep === 2) {
      testConnection();
    } else if (currentStep === 3) {
      finish();
    }
  }
});

// Focus the input on step 2
document.addEventListener('DOMContentLoaded', () => {
  // Check if we're on step 2 and focus the input
  const observer = new MutationObserver(() => {
    if (currentStep === 2) {
      const input = document.getElementById('gateway-url-input');
      if (input) {
        input.focus();
      }
    }
  });
  
  observer.observe(document.body, { childList: true, subtree: true });
});
