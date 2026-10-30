/**
 * SubWaste AI — Client-side Application Controller
 * Handles interactive prediction form, presets, batch CSV auditing, and live health metrics.
 */

// Presets data for quick testing
const PRESETS = {
  adobe: {
    subscription_name: "Adobe Creative Cloud",
    category: "Productivity",
    plan_tier: "Premium",
    monthly_fee: 54.99,
    billing_cycle: "Monthly",
    payment_method: "Credit Card",
    auto_renew: 1,
    usage_frequency_per_month: 0,
    hours_used_per_month: 0.0,
    days_since_last_login: 48,
    support_tickets_last_3m: 0,
    number_of_profiles_or_devices: 1,
    subscription_age_months: 14
  },
  spotify: {
    subscription_name: "Spotify Premium",
    category: "Streaming",
    plan_tier: "Standard",
    monthly_fee: 10.99,
    billing_cycle: "Monthly",
    payment_method: "Debit Card",
    auto_renew: 1,
    usage_frequency_per_month: 25,
    hours_used_per_month: 42.0,
    days_since_last_login: 1,
    support_tickets_last_3m: 0,
    number_of_profiles_or_devices: 2,
    subscription_age_months: 20
  },
  gym: {
    subscription_name: "Gold's Gym Membership",
    category: "Fitness",
    plan_tier: "Premium",
    monthly_fee: 49.99,
    billing_cycle: "Monthly",
    payment_method: "Credit Card",
    auto_renew: 1,
    usage_frequency_per_month: 1,
    hours_used_per_month: 0.5,
    days_since_last_login: 62,
    support_tickets_last_3m: 0,
    number_of_profiles_or_devices: 1,
    subscription_age_months: 9
  },
  cloud: {
    subscription_name: "Google One 2TB",
    category: "Cloud Storage",
    plan_tier: "Premium",
    monthly_fee: 9.99,
    billing_cycle: "Monthly",
    payment_method: "PayPal",
    auto_renew: 1,
    usage_frequency_per_month: 1,
    hours_used_per_month: 0.2,
    days_since_last_login: 38,
    support_tickets_last_3m: 0,
    number_of_profiles_or_devices: 1,
    subscription_age_months: 18
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initializeTabs();
  initializeSliderSync();
  initializeForm();
  initializeDropZone();
  loadSystemHealth();
  loadModelMetadata();
});

// Toast notification helper
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  
  const icon = type === "success" ? "✓" : type === "error" ? "⚠" : "ℹ";
  toast.innerHTML = `<span style="font-weight: bold; font-size: 1.1rem;">${icon}</span> <span>${message}</span>`;
  
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Navigation Tabs
function initializeTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const target = tab.dataset.target;
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
      });
      document.getElementById(target).classList.add("active");
    });
  });
}

// Synchronize Sliders with Numeric Inputs
function initializeSliderSync() {
  const pairs = [
    { slider: "daysInactiveSlider", input: "daysInactiveInput", label: "daysInactiveLabel" },
    { slider: "hoursUsedSlider", input: "hoursUsedInput", label: "hoursUsedLabel" },
    { slider: "activeDaysSlider", input: "activeDaysInput", label: "activeDaysLabel" }
  ];

  pairs.forEach(({ slider, input, label }) => {
    const s = document.getElementById(slider);
    const i = document.getElementById(input);
    const l = document.getElementById(label);

    s.addEventListener("input", (e) => {
      i.value = e.target.value;
      if (l) l.textContent = e.target.value;
    });

    i.addEventListener("input", (e) => {
      s.value = e.target.value;
      if (l) l.textContent = e.target.value;
    });
  });
}

// Apply Form Presets
function applyPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  document.getElementById("subName").value = data.subscription_name;
  document.getElementById("category").value = data.category;
  document.getElementById("planTier").value = data.plan_tier;
  document.getElementById("monthlyFee").value = data.monthly_fee;
  document.getElementById("billingCycle").value = data.billing_cycle;
  document.getElementById("autoRenew").value = data.auto_renew;
  document.getElementById("devices").value = data.number_of_profiles_or_devices;

  // Sliders & inputs
  document.getElementById("daysInactiveSlider").value = data.days_since_last_login;
  document.getElementById("daysInactiveInput").value = data.days_since_last_login;
  document.getElementById("daysInactiveLabel").textContent = data.days_since_last_login;

  document.getElementById("hoursUsedSlider").value = data.hours_used_per_month;
  document.getElementById("hoursUsedInput").value = data.hours_used_per_month;
  document.getElementById("hoursUsedLabel").textContent = data.hours_used_per_month;

  document.getElementById("activeDaysSlider").value = data.usage_frequency_per_month;
  document.getElementById("activeDaysInput").value = data.usage_frequency_per_month;
  document.getElementById("activeDaysLabel").textContent = data.usage_frequency_per_month;

  showToast(`Loaded preset: ${data.subscription_name}`, "info");
}

// Form Submission & Single Predict
function initializeForm() {
  const form = document.getElementById("auditorForm");
  const submitBtn = document.getElementById("auditSubmitBtn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      subscription_name: document.getElementById("subName").value.trim(),
      category: document.getElementById("category").value,
      plan_tier: document.getElementById("planTier").value,
      monthly_fee: parseFloat(document.getElementById("monthlyFee").value),
      billing_cycle: document.getElementById("billingCycle").value,
      payment_method: "Credit Card",
      auto_renew: parseInt(document.getElementById("autoRenew").value),
      usage_frequency_per_month: parseInt(document.getElementById("activeDaysInput").value),
      hours_used_per_month: parseFloat(document.getElementById("hoursUsedInput").value),
      days_since_last_login: parseInt(document.getElementById("daysInactiveInput").value),
      support_tickets_last_3m: 0,
      number_of_profiles_or_devices: parseInt(document.getElementById("devices").value),
      subscription_age_months: 12
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>Auditing with ML Pipeline...</span>`;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction request failed.");
      }

      const result = await response.json();
      renderPredictionResult(result);
      showToast(`Audit complete for ${result.subscription_name}`, "success");
    } catch (err) {
      showToast(`Error: ${err.message}`, "error");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>Audit Subscription Now</span>`;
    }
  });
}

// Render Single Prediction in UI
function renderPredictionResult(data) {
  const card = document.getElementById("resultCard");
  const tag = document.getElementById("resultRiskTag");
  const title = document.getElementById("resultSubTitle");
  const probVal = document.getElementById("resultProbValue");
  const barFill = document.getElementById("resultBarFill");
  const recText = document.getElementById("resultRecommendation");
  const unitCost = document.getElementById("resultUnitCost");
  const inactRatio = document.getElementById("resultInactivityRatio");
  const annualCost = document.getElementById("resultAnnualCost");

  title.textContent = `${data.subscription_name} ($${data.monthly_fee.toFixed(2)}/mo)`;
  recText.textContent = data.recommendation;

  unitCost.textContent = `$${data.engineered_metrics.cost_per_active_hour.toFixed(2)}/hr`;
  inactRatio.textContent = `${data.engineered_metrics.inactivity_ratio.toFixed(2)}x cycle`;
  annualCost.textContent = `$${data.engineered_metrics.annual_projected_cost.toFixed(2)}/yr`;

  const prob = data.waste_probability;
  probVal.textContent = `${prob.toFixed(1)}%`;
  barFill.style.width = `${prob}%`;

  // Dynamic status styling
  card.classList.remove("wasteful", "utilized");
  tag.classList.remove("high", "low", "moderate");
  barFill.classList.remove("high", "low", "moderate");

  if (data.is_wasteful === 1) {
    card.classList.add("wasteful");
    tag.classList.add("high");
    barFill.classList.add("high");
    tag.textContent = "HIGH RISK (Wasteful)";
  } else if (prob >= 35) {
    tag.classList.add("moderate");
    barFill.classList.add("moderate");
    tag.textContent = "MODERATE RISK";
  } else {
    card.classList.add("utilized");
    tag.classList.add("low");
    barFill.classList.add("low");
    tag.textContent = "LOW RISK (Well Utilized)";
  }
}

// Drag & Drop CSV Audit
function initializeDropZone() {
  const dropZone = document.getElementById("csvDropZone");
  const fileInput = document.getElementById("csvFileInput");

  ["dragenter", "dragover"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
  });
}

// Upload CSV to /predict/csv
async function handleFileUpload(file) {
  if (!file.name.endsWith(".csv")) {
    showToast("Please upload a valid .csv file.", "error");
    return;
  }

  showToast(`Uploading and auditing ${file.name}...`, "info");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict/csv", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "CSV processing failed.");
    }

    const batchData = await res.json();
    renderBatchResults(batchData);
    showToast(`Successfully audited ${batchData.total_subscriptions} subscriptions!`, "success");
  } catch (err) {
    showToast(`CSV Audit Failed: ${err.message}`, "error");
  }
}

// Render Batch CSV Audit Results
function renderBatchResults(data) {
  // Update Top Hero KPIs
  document.getElementById("kpiTotalSubs").textContent = data.total_subscriptions;
  document.getElementById("kpiMonthlySpend").textContent = `$${data.total_monthly_spend.toFixed(2)}`;
  document.getElementById("kpiWasteCount").textContent = `${data.wasteful_subscriptions_count} (${data.waste_rate_percentage}%)`;
  document.getElementById("kpiAnnualSavings").textContent = `$${data.total_potential_annual_savings.toFixed(2)}`;

  // Populate Table
  const tbody = document.getElementById("batchTableBody");
  tbody.innerHTML = "";

  data.results.forEach(sub => {
    const tr = document.createElement("tr");

    const badgeClass = sub.is_wasteful === 1 ? "high" : (sub.waste_probability >= 35 ? "moderate" : "low");
    const badgeText = sub.is_wasteful === 1 ? "Wasteful" : "Utilized";

    tr.innerHTML = `
      <td><strong>${sub.subscription_name}</strong></td>
      <td>$${sub.monthly_fee.toFixed(2)}</td>
      <td>
        <span class="risk-tag ${badgeClass}" style="font-size: 0.72rem; padding: 0.2rem 0.5rem;">
          ${badgeText} (${sub.waste_probability.toFixed(0)}%)
        </span>
      </td>
      <td>$${sub.engineered_metrics.cost_per_active_hour.toFixed(2)}/hr</td>
      <td>$${sub.engineered_metrics.annual_projected_cost.toFixed(2)}</td>
      <td style="max-width: 320px; font-size: 0.8rem; color: #cbd5e1;">${sub.recommendation}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("batchTableContainer").style.display = "block";
}

// Built-in Sample CSV Generator & Downloader
function downloadSampleCSV() {
  const sampleCSV = 
`subscription_name,category,plan_tier,monthly_fee,billing_cycle,payment_method,auto_renew,usage_frequency_per_month,hours_used_per_month,days_since_last_login,support_tickets_last_3m,number_of_profiles_or_devices,subscription_age_months
Adobe Creative Cloud,Productivity,Premium,54.99,Monthly,Credit Card,1,0,0.0,52,0,1,16
Spotify Family,Streaming,Standard,16.99,Monthly,Debit Card,1,26,45.0,1,0,4,24
Gold's Gym Membership,Fitness,Premium,49.99,Monthly,Credit Card,1,1,0.5,60,0,1,10
Netflix Premium,Streaming,Premium,22.99,Monthly,PayPal,1,18,32.0,2,0,3,30
Coursera Plus,Education,Standard,39.00,Monthly,Credit Card,1,0,0.0,45,0,1,6
PlayStation Plus,Gaming,Standard,14.99,Monthly,Credit Card,1,14,18.0,4,0,1,12`;

  const blob = new Blob([sampleCSV], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sample_subscriptions_audit.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  showToast("Sample CSV downloaded! Drag and drop it back into the upload zone to test.", "info");
}

// Live Health Check
async function loadSystemHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const dot = document.getElementById("healthDot");
    const text = document.getElementById("healthText");

    if (data.status === "healthy" && data.model_loaded) {
      dot.className = "pulse-dot";
      text.textContent = "Pipeline Active (RandomForest)";
    } else {
      dot.className = "pulse-dot degraded";
      text.textContent = "Model Degraded";
    }
  } catch (e) {
    document.getElementById("healthDot").className = "pulse-dot degraded";
    document.getElementById("healthText").textContent = "API Offline";
  }
}

// Model Metadata & Feature Importance Loader
async function loadModelMetadata() {
  try {
    const res = await fetch("/model/metadata");
    if (!res.ok) return;
    const meta = await res.json();

    document.getElementById("metaModelType").textContent = meta.model_type || "RandomForestClassifier";
    document.getElementById("metaAccuracy").textContent = `${((meta.evaluation_metrics.accuracy || 0) * 100).toFixed(1)}%`;
    document.getElementById("metaPrecision").textContent = `${((meta.evaluation_metrics.precision || 0) * 100).toFixed(1)}%`;
    document.getElementById("metaRecall").textContent = `${((meta.evaluation_metrics.recall || 0) * 100).toFixed(1)}%`;
    document.getElementById("metaF1").textContent = `${((meta.evaluation_metrics.f1_score || 0) * 100).toFixed(1)}%`;

    // Static display of top importance ranking from evaluation
    const features = [
      { name: "hours_used_per_month", weight: 16.0 },
      { name: "days_since_last_login", weight: 14.7 },
      { name: "cost_per_active_day", weight: 11.3 },
      { name: "usage_frequency_per_month", weight: 11.2 },
      { name: "utilization_rate", weight: 10.5 },
      { name: "monthly_fee", weight: 9.8 }
    ];

    const container = document.getElementById("featureBarsContainer");
    container.innerHTML = "";
    features.forEach(f => {
      const row = document.createElement("div");
      row.className = "feature-bar-row";
      row.innerHTML = `
        <div class="feature-bar-meta">
          <span>${f.name}</span>
          <span>${f.weight.toFixed(1)}%</span>
        </div>
        <div class="feature-progress-bg">
          <div class="feature-progress-fill" style="width: ${f.weight * 5}%;"></div>
        </div>
      `;
      container.appendChild(row);
    });
  } catch (e) {
    console.warn("Could not load model metadata:", e);
  }
}
