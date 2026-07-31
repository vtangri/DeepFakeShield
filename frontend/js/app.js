/**
 * DeepFakeShield AI - Premium Frontend Application
 * Features: 3D Tilt, Intersection Observer, Dynamic Charts
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:8000/api/v1' 
    : `${window.location.origin}/api/v1`;

// --- VISUAL EFFECTS ---

class VFX {
    static init() {
        this.initTilt();
        this.initScrollReveal();
        this.initMagneticButtons();
    }

    // 3D Tilt Effect for Glass Cards
    static initTilt() {
        document.querySelectorAll('.glass-card, .feature-card, .step-card, .verdict-card, .report-item').forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = ((y - centerY) / centerY) * -5;
                const rotateY = ((x - centerX) / centerX) * 5;
                
                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
                
                // Dynamic Highlight
                const sheenX = (x / rect.width) * 100;
                const sheenY = (y / rect.height) * 100;
                card.style.background = `radial-gradient(circle at ${sheenX}% ${sheenY}%, rgba(255,255,255,0.08), rgba(255,255,255,0.03) 40%)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
                card.style.background = 'var(--bg-panel)';
            });
        });
    }

    // Scroll Reveal Animation
    static initScrollReveal() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, { threshold: 0.1 });

        document.querySelectorAll('.feature-card, .step-card, .hero-content > *').forEach((el, index) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = `all 0.6s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.05}s`;
            observer.observe(el);
        });
    }
    
    // Auto-init on new content
    static refresh() {
        this.initTilt();
    }

    // Magnetic Buttons Effect
    static initMagneticButtons() {
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('mousemove', (e) => {
                const rect = btn.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                
                btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
            });

            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'translate(0, 0)';
            });
        });
    }
}

// --- STATE MANAGEMENT ---

const state = {
    user: null,
    token: localStorage.getItem('token'),
    currentJobId: null,
    selectedFile: null
};

// --- API CLIENT ---

const api = {
    async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        console.log(`API Request: ${options.method || 'GET'} ${url}`);
        
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
        
        try {
            const response = await fetch(url, { ...options, headers });
            // Handle 401 Unauthorized
            if (response.status === 401) {
                logout();
                throw new Error('Session expired');
            }
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Request failed');
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    async login(email, password) {
        const body = JSON.stringify({ email, password });
        
        const response = await fetch(`${API_BASE}/auth/login`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body 
        });
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.detail || 'Login failed');
        state.token = data.access_token;
        localStorage.setItem('token', state.token);
        
        // Fetch user details immediately after login
        await this.getUser();
        return data;
    },
    
    async register(email, password, fullName) {
         return this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ email, password, full_name: fullName })
        });
    },

    async uploadMedia(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // Remove Content-Type header to let browser set boundary
        const response = await fetch(`${API_BASE}/media/upload`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${state.token}` },
            body: formData
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Upload failed');
        return data;
    },
    
    async startAnalysis(mediaId) {
        return this.request('/analysis/start', {
            method: 'POST',
            body: JSON.stringify({ media_id: mediaId })
        });
    },
    
    async getJobStatus(jobId) { return this.request(`/analysis/${jobId}/status`); },
    async getJobResult(jobId) { return this.request(`/analysis/${jobId}/result`); },
    
    async getUser() { 
        try {
            const user = await this.request('/auth/me'); 
            state.user = user;
            return user;
        } catch (e) {
            state.user = null;
            throw e;
        }
    },
    
    async listMedia() { return this.request('/media/'); },
    async listReports() { return this.request('/reports/'); },
    
    // Evidence & Reports
    async getEvidenceTimeline(jobId) { return this.request(`/analysis/${jobId}/evidence/timeline`); },
    
    async downloadPDF(jobId) {
        const url = `${API_BASE}/reports/${jobId}/report.pdf`;
        const response = await fetch(url, {
            headers: { 'Authorization': `Bearer ${state.token}` }
        });
        if (!response.ok) throw new Error('PDF download failed');
        
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `deepfakeshield_report_${jobId.slice(0, 8)}.pdf`;
        link.click();
        URL.revokeObjectURL(link.href);
    }
};

// --- CORE LOGIC ---

function navigateTo(pageId) {
    // Fade out current
    const current = document.querySelector('.page.active');
    if(current) {
        current.style.opacity = '0';
        current.style.transform = 'translateY(-20px)';
        setTimeout(() => current.classList.remove('active'), 300);
    }

    // Fade in new
    setTimeout(() => {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        const newPage = document.getElementById(`${pageId}-page`);
        if(newPage) {
            newPage.classList.add('active');
            requestAnimationFrame(() => {
                newPage.style.opacity = '1';
                newPage.style.transform = 'translateY(0)';
            });
            
            // Page specific logic
            if (pageId === 'reports') loadReports();
            if (pageId === 'settings') loadSettings();
        }
    }, 300);

    // Update Nav
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelector(`[data-page="${pageId}"]`)?.classList.add('active');
    
    // Close dropdown if open
    document.getElementById('userDropdown').classList.remove('active');
    
    VFX.refresh();
}

// File Upload Handler with Drag Visuals
function initFileUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => handleFiles(e.target.files));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    zone.addEventListener('dragenter', () => zone.classList.add('dragover'));
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        zone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    function handleFiles(files) {
        if(files.length > 0) {
            const file = files[0];
            state.selectedFile = file;
            
            // Visual Update
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = formatBytes(file.size);
            document.getElementById('uploadPreview').style.display = 'block';
            document.querySelector('.upload-content').style.display = 'none';
            
            // Auto scroll to preview
            document.getElementById('uploadPreview').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Remove file handler
        document.getElementById('removeFile').addEventListener('click', (e) => {
            e.stopPropagation();
            state.selectedFile = null;
            document.getElementById('fileInput').value = '';
            document.getElementById('uploadPreview').style.display = 'none';
            document.querySelector('.upload-content').style.display = 'block';
        });
    }
}

// Analysis Workflow
async function startAnalysis() {
    if (!state.selectedFile || !state.token) {
        if(!state.token) showAuthModal();
        else showToast('Please upload a file first.', 'info');
        return;
    }

    navigateTo('processing');
    resetAnalysisState(); // Reset activity log and stage cards
    
    try {
        updateStatus('validating', 0, 'Uploading Media...');
        const upload = await api.uploadMedia(state.selectedFile);
        
        updateStatus('queued', 10, 'Initializing AI Models...');
        const job = await api.startAnalysis(upload.id);
        
        pollStatus(job.job_id);
    } catch (e) {
        showToast(e.message, 'error');
        navigateTo('upload');
    }
}

async function pollStatus(jobId) {
    const poll = async () => {
        try {
            const status = await api.getJobStatus(jobId);
            
            // Normalize stage to lowercase
            const stage = (status.stage || 'pending').toLowerCase();
            
            // Map stage to progress - include all possible backend stages
            const progressMap = {
                'pending': 5,
                'validating': 15,
                'extracting': 30,
                'transcribing': 45,
                'infer_video': 55,
                'infer_audio': 65,
                'lipsync': 75,
                'analyzing': 70,
                'fusion': 85,
                'report': 95,
                'reporting': 95,
                'done': 100
            };
            
            // Friendly stage names for display
            const stageNames = {
                'pending': 'Initializing...',
                'validating': 'Validating file...',
                'extracting': 'Extracting frames & audio...',
                'transcribing': 'Transcribing audio...',
                'infer_video': 'Analyzing video frames...',
                'infer_audio': 'Analyzing audio patterns...',
                'lipsync': 'Checking lip synchronization...',
                'analyzing': 'Running AI models...',
                'fusion': 'Fusing predictions...',
                'report': 'Generating report...',
                'reporting': 'Generating report...',
                'done': 'Complete!'
            };
            
            const pct = progressMap[stage] || 50;
            const displayName = stageNames[stage] || `Processing: ${stage}...`;
            updateStatus(stage, pct, displayName);
            
            if (status.status.toLowerCase() === 'done' || stage === 'done') {
                const result = await api.getJobResult(jobId);
                showResults(result);
            } else if (status.status.toLowerCase() === 'failed') {
                showToast('Analysis failed: ' + status.error_message, 'error');
                navigateTo('upload');
            } else {
                // Ultra-fast polling for instant progress updates
                setTimeout(poll, 200);
            }
        } catch (e) {
            console.error(e);
            setTimeout(poll, 500);
        }
    };
    poll();
}

// Track analysis start time for activity log
let analysisStartTime = null;

function updateStatus(stage, percent, text) {
    const bar = document.getElementById('progressFill');
    const txt = document.getElementById('progressText');
    const label = document.getElementById('processingStage');
    
    // Initialize start time
    if (!analysisStartTime) analysisStartTime = Date.now();
    
    bar.style.width = `${percent}%`;
    txt.textContent = `${percent}%`;
    label.textContent = text;
    
    const stages = ['validating', 'extracting', 'transcribing', 'analyzing', 'reporting'];
    const currentIndex = stages.indexOf(stage);
    
    // Update stage cards
    if (currentIndex > -1) {
        stages.forEach((s, idx) => {
            const el = document.getElementById(`stage-${s}`);
            if (!el) return;
            
            if (idx < currentIndex) {
                el.classList.add('completed');
                el.classList.remove('active');
            } else if (idx === currentIndex) {
                el.classList.add('active');
                el.classList.remove('completed');
            } else {
                el.classList.remove('active', 'completed');
            }
        });
    }
    
    // Update activity log with detailed messages
    addActivityLog(stage, currentIndex);
}

// Activity log messages for each stage (matching all backend stages)
const stageActivities = {
    validating: [
        'Checking file integrity...',
        'Extracting metadata (codec, resolution, fps)...'
    ],
    extracting: [
        'Sampling keyframes at 1fps...',
        'Extracting audio track (PCM 16-bit)...'
    ],
    transcribing: [
        'Loading Whisper ASR model...',
        'Transcribing audio to text...'
    ],
    infer_video: [
        'Running ViT-B/16 face classifier...',
        'Detecting facial boundary artifacts...'
    ],
    infer_audio: [
        'Analyzing audio spectrogram patterns...',
        'Checking for voice cloning signatures...'
    ],
    lipsync: [
        'Computing lip-sync correlation matrix...',
        'Measuring audio-visual alignment...'
    ],
    fusion: [
        'Fusing multimodal predictions...',
        'Applying ensemble weighting...'
    ],
    report: [
        'Generating forensic summary...',
        'Analysis complete! ✓'
    ],
    done: [
        'Results ready!'
    ]
};

let lastStage = null;
let activityIndex = 0;

function addActivityLog(stage, stageIndex) {
    const logContainer = document.getElementById('activityLog');
    if (!logContainer) return;
    
    // Reset on new stage
    if (stage !== lastStage) {
        lastStage = stage;
        activityIndex = 0;
    }
    
    const activities = stageActivities[stage] || [];
    if (activityIndex >= activities.length) return;
    
    // Calculate elapsed time
    const elapsed = ((Date.now() - analysisStartTime) / 1000).toFixed(1);
    const timeStr = elapsed.padStart(5, '0');
    
    // Add new activity item
    const item = document.createElement('div');
    item.className = 'activity-item';
    item.innerHTML = `
        <span class="activity-time">${timeStr}s</span>
        <span class="activity-text">${activities[activityIndex]}</span>
    `;
    logContainer.appendChild(item);
    
    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
    
    activityIndex++;
}

// Reset analysis state when navigating to processing
function resetAnalysisState() {
    analysisStartTime = null;
    lastStage = null;
    activityIndex = 0;
    
    const logContainer = document.getElementById('activityLog');
    if (logContainer) {
        logContainer.innerHTML = `
            <div class="activity-item">
                <span class="activity-time">00:00</span>
                <span class="activity-text">Starting analysis pipeline...</span>
            </div>
        `;
    }
    
    // Reset all stage cards
    document.querySelectorAll('.stage-card').forEach(el => {
        el.classList.remove('active', 'completed');
    });
}

function showResults(data) {
    // Store job ID for later use (PDF download)
    state.currentJobId = data.job_id;
    
    // Populate Score
    const score = Math.round((data.overall_score || 0) * 100);
    document.getElementById('scoreValue').textContent = `${score}%`;
    
    // Update Ring
    const circle = document.getElementById('scoreCircle');
    const offset = 283 - (283 * (data.overall_score || 0));
    circle.style.strokeDashoffset = offset;
    
    // Update modality scores
    updateModalityScores(data);
    
    // Render timeline with segments
    renderEvidenceTimeline(data.segments || []);
    
    // Verdict Text
    const verdict = document.querySelector('.verdict-label');
    const desc = document.querySelector('.verdict-description');
    
    if(score < 30) {
        verdict.textContent = 'LIKELY AUTHENTIC';
        verdict.style.color = 'var(--success)';
        circle.style.stroke = 'var(--success)';
        desc.textContent = 'No significant manipulation detected.';
        document.querySelector('.verdict-icon').innerHTML = '✓';
        document.querySelector('.verdict-icon').className = 'verdict-icon authentic';
        document.getElementById('verdictCard').style.borderColor = 'rgba(16, 185, 129, 0.2)';
        document.getElementById('verdictCard').style.background = 'linear-gradient(145deg, rgba(16, 185, 129, 0.1), transparent)';

    } else if(score < 70) {
        verdict.textContent = 'SUSPICIOUS';
        verdict.style.color = 'var(--warning)';
        circle.style.stroke = 'var(--warning)';
        desc.textContent = 'Some anomalies detected. Manual review recommended.';
        document.querySelector('.verdict-icon').innerHTML = '⚠';
        document.querySelector('.verdict-icon').className = 'verdict-icon';
        document.querySelector('.verdict-icon').style.background = 'var(--warning)';
        document.getElementById('verdictCard').style.borderColor = 'rgba(245, 158, 11, 0.2)';
        document.getElementById('verdictCard').style.background = 'linear-gradient(145deg, rgba(245, 158, 11, 0.1), transparent)';

    } else {
        verdict.textContent = 'HIGHLY LIKELY FAKE';
        verdict.style.color = 'var(--error)';
        circle.style.stroke = 'var(--error)';
        desc.textContent = 'Strong evidence of manipulation found.';
        document.querySelector('.verdict-icon').innerHTML = '✕';
        document.querySelector('.verdict-icon').className = 'verdict-icon fake';
        document.getElementById('verdictCard').style.borderColor = 'rgba(239, 68, 68, 0.2)';
        document.getElementById('verdictCard').style.background = 'linear-gradient(145deg, rgba(239, 68, 68, 0.1), transparent)';
    }

    // Show media preview (video or image)
    showMediaPreview();

    navigateTo('results');
    VFX.refresh();
}

function showMediaPreview() {
    const previewSection = document.getElementById('mediaPreviewSection');
    const previewVideo = document.getElementById('mediaPreviewVideo');
    const previewImage = document.getElementById('mediaPreviewImage');
    
    if (!previewSection) return;

    if (!state.selectedFile) {
        // If we don't have the file (e.g. from history), hide preview section
        // or show a generic placeholder if we had thumbnails (not implemented yet)
        previewSection.style.display = 'none';
        return;
    }
    
    // Create object URL from the selected file
    const fileUrl = URL.createObjectURL(state.selectedFile);
    const fileType = state.selectedFile.type;
    
    if (fileType.startsWith('video/')) {
        // Show video player
        previewVideo.src = fileUrl;
        previewVideo.style.display = 'block';
        previewImage.style.display = 'none';
    } else if (fileType.startsWith('image/')) {
        // Show image
        previewImage.src = fileUrl;
        previewImage.style.display = 'block';
        previewVideo.style.display = 'none';
    } else if (fileType.startsWith('audio/')) {
        // For audio, show a styled audio element
        previewVideo.src = fileUrl;
        previewVideo.style.display = 'block';
        previewVideo.style.height = '60px';
        previewImage.style.display = 'none';
    } else {
        previewSection.style.display = 'none';
        return;
    }
    
    previewSection.style.display = 'block';
}

function updateModalityScores(data) {
    // Get results object (contains all the detailed data)
    const results = data.results || {};
    const video = results.video || {};
    const audio = results.audio || {};
    const lipsync = results.lipsync || {};
    const metadata = results.metadata || {};
    const techSummary = results.technical_summary || {};
    const artifacts = video.artifacts || {};
    
    // Video Analysis
    const hasVideo = video.score !== null && video.score !== undefined;
    const videoScore = hasVideo ? Math.round(video.score * 100) : null;
    const videoBar = document.getElementById('videoScoreBar');
    const videoText = document.getElementById('videoScoreText');
    if (videoBar) {
        if (hasVideo) {
            videoBar.style.width = `${videoScore}%`;
            videoBar.className = `score-fill ${videoScore < 30 ? 'safe' : videoScore < 70 ? 'warning' : 'danger'}`;
        } else {
            videoBar.style.width = `0%`;
            videoBar.className = `score-fill disabled`;
        }
    }
    if (videoText) videoText.textContent = hasVideo ? `${videoScore}%` : 'N/A';
    
    // Video Confidence
    const videoConf = document.getElementById('videoConfidence');
    if (videoConf) {
        videoConf.textContent = hasVideo && video.confidence != null ? 
            `${Math.round(video.confidence * 100)}%` : 'N/A';
    }
    
    // Video Details
    setText('framesAnalyzed', hasVideo ? (video.frames_analyzed || '--') : 'N/A');
    setText('facesDetected', hasVideo ? (video.faces_detected || '--') : 'N/A');
    setText('manipulationType', hasVideo ? (video.manipulation_type || 'None') : 'N/A');
    setText('blendingScore', hasVideo && video.frame_analysis?.blending_score != null ? 
        `${(video.frame_analysis.blending_score * 100).toFixed(1)}%` : 'N/A');
    
    // Video Description
    const videoDesc = document.getElementById('videoDescription');
    if (videoDesc) {
        if (!hasVideo) {
            videoDesc.textContent = 'Video analysis not applicable for this media type.';
        } else if (videoScore < 30) {
            videoDesc.textContent = 'No facial manipulation artifacts detected in analyzed frames.';
        } else if (videoScore < 70) {
            videoDesc.textContent = `Potential ${video.manipulation_type || 'manipulation'} detected. Manual review recommended.`;
        } else {
            videoDesc.textContent = `High confidence ${video.manipulation_type || 'deepfake'} detected via ${video.manipulation_method || 'AI synthesis'}.`;
        }
    }
    
    // Audio Analysis
    const hasAudio = audio.score !== null && audio.score !== undefined;
    const audioScore = hasAudio ? Math.round(audio.score * 100) : null;
    const audioBar = document.getElementById('audioScoreBar');
    const audioText = document.getElementById('audioScoreText');
    if (audioBar) {
        if (hasAudio) {
            audioBar.style.width = `${audioScore}%`;
            audioBar.className = `score-fill ${audioScore < 30 ? 'safe' : audioScore < 70 ? 'warning' : 'danger'}`;
        } else {
            audioBar.style.width = `0%`;
            audioBar.className = `score-fill disabled`;
        }
    }
    if (audioText) audioText.textContent = hasAudio ? `${audioScore}%` : 'N/A';
    
    // Audio Confidence
    const audioConf = document.getElementById('audioConfidence');
    if (audioConf) {
        audioConf.textContent = hasAudio && audio.confidence != null ? 
            `${Math.round(audio.confidence * 100)}%` : 'N/A';
    }
    
    // Audio Details
    setText('voiceCloning', hasAudio ? (audio.voice_cloning_detected ? 
        `Detected (${audio.cloning_method || 'Unknown'})` : 'Not Detected') : 'N/A');
    setText('mfccScore', hasAudio && audio.spectral_analysis?.mfcc_anomaly_score != null ?
        `${(audio.spectral_analysis.mfcc_anomaly_score * 100).toFixed(1)}%` : 'N/A');
    setText('naturalness', hasAudio && audio.voice_identity?.naturalness_score != null ?
        `${Math.round(audio.voice_identity.naturalness_score * 100)}%` : 'N/A');
    setText('formantStatus', hasAudio ? (audio.spectral_analysis?.formant_consistency || '--') : 'N/A');
    
    // Linguistic Analysis
    const ling = results.linguistic_analysis || {};
    const lingPatterns = ling.suspicious_patterns || {};
    
    setText('aiTextProb', hasAudio && ling.generated_text_probability != null ? `${Math.round(ling.generated_text_probability * 100)}%` : 'N/A');
    setText('templatedSpeech', hasAudio ? (lingPatterns.templated_speech ? '⚠️ Yes' : 'No') : 'N/A');
    setText('repetitionStatus', hasAudio ? (lingPatterns.unnatural_repetition ? '⚠️ Detected' : 'Normal') : 'N/A');
    
    // Audio Description
    const audioDesc = document.getElementById('audioDescription');
    if (audioDesc) {
        if (!hasAudio) {
            audioDesc.textContent = 'Audio analysis not applicable (no audio track or still image).';
        } else if (audioScore < 30) {
            audioDesc.textContent = 'Voice patterns consistent with natural speech.';
        } else if (audioScore < 70) {
            audioDesc.textContent = 'Some audio anomalies detected in spectral analysis.';
        } else {
            audioDesc.textContent = `Voice cloning detected (${audio.cloning_method || 'AI synthesis'}). Unnatural formants.`;
        }
    }
    
    // Lip-Sync Analysis
    const hasLipsync = lipsync.score !== null && lipsync.score !== undefined;
    const lipsyncScore = hasLipsync ? Math.round(lipsync.score * 100) : null;
    const lipsyncBar = document.getElementById('lipsyncScoreBar');
    const lipsyncText = document.getElementById('lipsyncScoreText');
    if (lipsyncBar) {
        if (hasLipsync) {
            lipsyncBar.style.width = `${lipsyncScore}%`;
            lipsyncBar.className = `score-fill ${lipsyncScore < 30 ? 'safe' : lipsyncScore < 70 ? 'warning' : 'danger'}`;
        } else {
            lipsyncBar.style.width = `0%`;
            lipsyncBar.className = `score-fill disabled`;
        }
    }
    if (lipsyncText) lipsyncText.textContent = hasLipsync ? `${lipsyncScore}%` : 'N/A';
    
    // Lip-Sync Confidence
    const lipsyncConf = document.getElementById('lipsyncConfidence');
    if (lipsyncConf) {
        lipsyncConf.textContent = hasLipsync && lipsync.confidence != null ? 
            `${Math.round(lipsync.confidence * 100)}%` : 'N/A';
    }
    
    // Lip-Sync Details
    setText('syncOffset', hasLipsync && lipsync.sync_offset_ms != null ? `${lipsync.sync_offset_ms}ms` : 'N/A');
    
    const corrVal = lipsync.correlation !== undefined ? lipsync.correlation : lipsync.correlation_score;
    setText('correlation', hasLipsync && corrVal != null ? 
        `${(corrVal * 100).toFixed(1)}%` : 'N/A');
    setText('phonemeAccuracy', hasLipsync && lipsync.phoneme_accuracy != null ?
        `${Math.round(lipsync.phoneme_accuracy * 100)}%` : 'N/A');
    setText('visemeMatch', hasLipsync && lipsync.viseme_match_rate != null ?
        `${Math.round(lipsync.viseme_match_rate * 100)}%` : 'N/A');
    
    // Lip-Sync Description
    const lipsyncDesc = document.getElementById('lipsyncDescription');
    if (lipsyncDesc) {
        if (!hasLipsync) {
            lipsyncDesc.textContent = 'Lip-sync analysis not applicable (requires video with audio and visible faces).';
        } else if (lipsyncScore < 30) {
            lipsyncDesc.textContent = 'Audio-visual synchronization within normal parameters.';
        } else if (lipsyncScore < 70) {
            lipsyncDesc.textContent = `Sync offset of ${lipsync.sync_offset_ms || 'N/A'}ms detected. May indicate dubbing.`;
        } else {
            lipsyncDesc.textContent = `Severe desync detected (${lipsync.sync_offset_ms || 'N/A'}ms). Lip movement doesn't match audio.`;
        }
    }
    
    // Artifacts Detection
    updateArtifact('artifactBoundary', !artifacts.boundary_artifacts);
    updateArtifact('artifactTemporal', !artifacts.temporal_inconsistency);
    
    // New Artifacts
    const freq = results.frequency_analysis || {};
    updateArtifact('artifactGan', !freq.gan_fingerprint_detected);
    updateArtifact('artifactSpectrum', freq.spectrum_consistency !== 'ABNORMAL');

    // Media Quality Card
    const quality = results.media_quality || {};
    setText('qualityScore', quality.overall_quality_score ? `${quality.overall_quality_score}/100` : '--');
    
    const blur = quality.blur_detection || {};
    const noise = quality.noise_level || {};
    const compression = quality.compression_analysis || {};
    
    setText('blurStatus', blur.is_blurry ? `Detected (${blur.blur_score})` : 'Clear');
    setText('noiseLevel', noise.snr_db ? `${noise.snr_db} dB` : '--');
    setText('compressionStatus', compression.double_compression_detected ? 'Double Compression' : 'Single Pass');
    setText('integrityStatus', metadata.file_hash ? 'Verified' : '--');

    // Technical Summary & Container Forensics
    const container = results.container_analysis || {};
    
    setText('modelsUsed', techSummary.models_used?.length ? 
        techSummary.models_used.length + ' models' : '--');
    
    setText('containerStatus', container.metadata_consistency || '--');
    setText('toolFingerprints', container.tool_fingerprints?.join(', ') || 'None detected');
    
    setText('mediaResolution', metadata.resolution || '--');
    setText('mediaCodec', metadata.codec || '--');
    setText('dateMismatch', container.modification_date_mismatch ? '⚠️ Mismatch' : 'Consistent');
}

// Helper to safely set text content
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// Helper to update artifact indicator
function updateArtifact(id, isSafe) {
    const el = document.getElementById(id);
    if (!el) return;
    const indicator = el.querySelector('.artifact-indicator');
    if (indicator) {
        indicator.textContent = isSafe ? '✓' : '✕';
        indicator.className = `artifact-indicator ${isSafe ? 'safe' : 'danger'}`;
    }
}

function renderEvidenceTimeline(segments) {
    const container = document.getElementById('timelineContainer');
    if (!container) return;
    
    // Duration in ms (default 15.5 seconds)
    const durationMs = 15500;
    
    if (!segments || segments.length === 0) {
        container.innerHTML = `
            <div class="timeline-empty">
                <span>✓</span>
                <p>No anomalies detected in timeline</p>
            </div>
        `;
        return;
    }
    
    // Build timeline HTML
    const markersHtml = segments.map(seg => {
        const leftPct = (seg.start_ms / durationMs) * 100;
        const widthPct = ((seg.end_ms - seg.start_ms) / durationMs) * 100;
        const typeClass = seg.segment_type || 'video';
        const startSec = (seg.start_ms / 1000).toFixed(1);
        const endSec = (seg.end_ms / 1000).toFixed(1);
        
        return `
            <div class="timeline-marker ${typeClass}" 
                 style="left: ${leftPct}%; width: ${widthPct}%;"
                 title="${seg.reason}\n${startSec}s - ${endSec}s (Score: ${Math.round(seg.score * 100)}%)">
            </div>
        `;
    }).join('');
    
    container.innerHTML = `
        <div class="timeline-track">
            <div class="timeline-baseline"></div>
            ${markersHtml}
        </div>
        <div class="timeline-labels">
            <span>0s</span>
            <span>${(durationMs / 2000).toFixed(1)}s</span>
            <span>${(durationMs / 1000).toFixed(1)}s</span>
        </div>
        <div class="timeline-legend">
            <div class="legend-item"><span class="legend-dot video"></span> Video</div>
            <div class="legend-item"><span class="legend-dot audio"></span> Audio</div>
            <div class="legend-item"><span class="legend-dot lipsync"></span> Lip-Sync</div>
        </div>
        <div class="timeline-segments-list">
            <h4>Flagged Segments</h4>
            ${segments.map(seg => `
                <div class="segment-item ${seg.segment_type}">
                    <span class="segment-time">${(seg.start_ms/1000).toFixed(1)}s - ${(seg.end_ms/1000).toFixed(1)}s</span>
                    <span class="segment-type">${seg.segment_type?.toUpperCase() || 'VIDEO'}</span>
                    <span class="segment-reason">${seg.reason || 'Anomaly detected'}</span>
                    <span class="segment-score">${Math.round(seg.score * 100)}%</span>
                </div>
            `).join('')}
        </div>
    `;
}

// --- REPORTS LOGIC ---

// --- REPORTS LOGIC ---

async function loadReports() {
    if (!state.token) {
        showAuthModal();
        navigateTo('home');
        return;
    }

    try {
        const reports = await api.listReports();
        
        // Render both views
        renderDashboard(reports);
        renderReportsList(reports);
        
    } catch (e) {
        console.error(e);
        document.getElementById('reportsList').innerHTML = '<p class="text-error" style="text-align:center">Failed to load reports.</p>';
    }
}

function renderDashboard(reports) {
    // 1. Calculate Stats
    const total = reports.length;
    if (total === 0) {
        document.getElementById('statTotal').textContent = '0';
        document.getElementById('statFakePercent').textContent = '0%';
        document.getElementById('statConfidence').textContent = '0%';
        return;
    }

    const fakes = reports.filter(r => (r.overall_score || 0) > 0.7).length;
    const suspicious = reports.filter(r => (r.overall_score || 0) > 0.3 && (r.overall_score || 0) <= 0.7).length;
    const real = total - fakes - suspicious;

    const fakePercent = Math.round((fakes / total) * 100);
    const avgConfidence = Math.round(reports.reduce((acc, r) => acc + (r.overall_score || 0), 0) / total * 100);

    // 2. Update Stat Cards
    document.getElementById('statTotal').textContent = total;
    document.getElementById('statFakePercent').textContent = `${fakePercent}%`;
    document.getElementById('statConfidence').textContent = `${avgConfidence}%`;

    // 3. Update Chart
    const pReal = Math.round((real / total) * 100);
    const pFake = Math.round((fakes / total) * 100);
    const pSuspicious = Math.round((suspicious / total) * 100);

    document.getElementById('barReal').style.width = `${pReal}%`;
    document.getElementById('valReal').textContent = `${pReal}%`;
    
    document.getElementById('barFake').style.width = `${pFake}%`;
    document.getElementById('valFake').textContent = `${pFake}%`;
    
    document.getElementById('barSuspicious').style.width = `${pSuspicious}%`;
    document.getElementById('valSuspicious').textContent = `${pSuspicious}%`;

    // 4. Recent Alerts
    const recentFakes = reports
        .filter(r => (r.overall_score || 0) > 0.5)
        .slice(0, 5);
        
    const alertsContainer = document.getElementById('recentAlerts');
    if (recentFakes.length === 0) {
        alertsContainer.innerHTML = '<p class="text-muted" style="text-align: center; margin-top: 2rem;">No recent threats detected</p>';
    } else {
        alertsContainer.innerHTML = recentFakes.map(r => {
            const type = r.media_type ? r.media_type.charAt(0).toUpperCase() + r.media_type.slice(1) : 'Media';
            const icon = r.media_type === 'audio' ? '🔊' : (r.media_type === 'image' ? '🖼️' : '🎬');
            return `
            <div class="threat-item" style="cursor: pointer;" data-job-id="${r.job_id}">
                <div class="threat-icon">${icon}</div>
                <div class="threat-info">
                    <h4>Fake ${type} Detected</h4>
                    <p>${new Date(r.created_at || r.generated_at).toLocaleDateString()} • Score: ${Math.round(r.overall_score * 100)}%</p>
                </div>
                <button class="btn btn-sm btn-ghost view-alert-btn" style="margin-left: auto; color: var(--accent);" data-job-id="${r.job_id}">View</button>
            </div>
        `}).join('');

        // Attach listeners to alerts
        alertsContainer.querySelectorAll('.threat-item').forEach(item => {
            item.addEventListener('click', async () => {
                const jobId = item.dataset.jobId;
                if (!jobId) return;
                try {
                    showToast('Loading analysis details...', 'info');
                    navigateTo('processing');
                    resetAnalysisState();
                    updateStatus('done', 100, 'Retrieving results...');
                    const result = await api.getJobResult(jobId);
                    showResults(result);
                } catch (err) {
                    console.error('Alert Click Error:', err);
                    showToast('Failed to load details: ' + err.message, 'error');
                    navigateTo('reports');
                }
            });
        });
    }
}

function renderReportsList(reports) {
    const container = document.getElementById('reportsList');
    const emptyState = document.getElementById('emptyReports');

    if (reports.length === 0) {
        container.innerHTML = '';
        container.style.display = 'none';
        emptyState.style.display = 'flex';
        return;
    }

    emptyState.style.display = 'none';
    container.style.display = 'flex'; // Flex column

    container.innerHTML = reports.map(r => {
        const score = Math.round(r.overall_score * 100);
        let statusClass = 'authentic';
        let statusText = 'AUTHENTIC';
        
        if (score > 70) { statusClass = 'fake'; statusText = 'FAKE'; }
        else if (score > 30) { statusClass = 'suspicious'; statusText = 'SUSPICIOUS'; }
        
        const statusStyle = statusClass === 'suspicious' ? 'background: rgba(245, 158, 11, 0.2); color: var(--warning);' : '';

        return `
        <div class="glass-card report-item" data-job-id="${r.job_id}">
            <div class="report-left">
                <div class="report-thumbnail">
                    ${r.media_type === 'audio' ? '🔊' : '🎬'}
                </div>
                <div class="report-meta">
                    <h4>Analysis #${r.job_id.slice(0, 8)}</h4>
                    <span class="report-date">${new Date(r.created_at).toLocaleString()}</span>
                </div>
            </div>
            <div class="report-right" style="display: flex; align-items: center; gap: 1.5rem;">
                    <div class="report-status ${statusClass}" style="${statusStyle}">
                    ${statusText} (${score}%)
                    </div>
                <button class="btn btn-ghost btn-sm download-pdf-btn" data-job-id="${r.job_id}">
                    Download PDF
                </button>
            </div>
        </div>
    `}).join('');
    
    // Attach event listeners to new buttons
    container.querySelectorAll('.download-pdf-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const jobId = btn.dataset.jobId;
            try {
                showToast('Generating PDF...', 'info');
                await api.downloadPDF(jobId);
                showToast('Report downloaded!', 'success');
            } catch (err) {
                console.error(err);
                showToast('Download failed: ' + err.message, 'error');
            }
        });
    });

    // Attach event listeners to report cards for details view
    container.querySelectorAll('.report-item').forEach(item => {
        item.addEventListener('click', async (e) => {
            // Don't trigger if clicking the download button
            if (e.target.closest('.download-pdf-btn')) return;
            
            const jobId = item.dataset.jobId;
            
            if (jobId) {
                try {
                    showToast('Loading analysis details...', 'info');
                    
                    // Switch to processing/loading state briefly
                    navigateTo('processing');
                    resetAnalysisState();
                    updateStatus('done', 100, 'Retrieving results...');
                    
                    const result = await api.getJobResult(jobId);
                    showResults(result);
                    
                } catch (err) {
                    console.error('History Click Error:', err);
                    showToast('Failed to load details: ' + err.message, 'error');
                    navigateTo('reports');
                }
            }
        });
    });

    // Refresh tilt effect for new items
    VFX.refresh();
}

function initReportsTabs() {
    const tabs = document.querySelectorAll('.report-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Toggle active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Toggle view
            const viewId = tab.dataset.view;
            document.querySelectorAll('.dashboard-view').forEach(v => v.classList.remove('active'));
            document.getElementById(`view-${viewId}`).classList.add('active');
        });
    });
}

// --- SETTINGS LOGIC ---
function loadSettings() {
    const settings = JSON.parse(localStorage.getItem('userSettings') || '{}');
    
    // Select both checkboxes and select inputs
    document.querySelectorAll('.toggle-switch input, .setting-select').forEach(input => {
        const key = input.dataset.setting;
        if (key && settings[key] !== undefined) {
            if(input.type === 'checkbox') input.checked = settings[key];
            else input.value = settings[key];
        }
        
        input.addEventListener('change', () => {
            if(input.type === 'checkbox') settings[key] = input.checked;
            else settings[key] = input.value;
            localStorage.setItem('userSettings', JSON.stringify(settings));
        });
    });
}


// --- UTILS ---

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024, sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAuthModal() {
    document.getElementById('authModal').classList.add('active');
}

function logout() {
    state.token = null;
    state.user = null;
    localStorage.removeItem('token');
    updateNavAuth();
    navigateTo('home');
    document.getElementById('userDropdown').classList.remove('active');
}

function updateNavAuth() {
    const authBtns = document.getElementById('authButtons');
    const userMenu = document.getElementById('userMenu');
    const userName = document.getElementById('userName');
    const userAvatar = document.getElementById('userAvatar');

    if(state.token && state.user) {
        authBtns.style.display = 'none';
        userMenu.style.display = 'block';
        
        // Update user info
        userName.textContent = state.user.full_name || 'User';
        const initials = (state.user.full_name || 'U').split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
        userAvatar.textContent = initials;
        
    } else {
        authBtns.style.display = 'flex';
        userMenu.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Icons based on type
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '✕';
    
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Remove after 3s
    setTimeout(() => {
        toast.classList.add('hiding');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3000);
}

function initPasswordToggles() {
    document.querySelectorAll('.password-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const input = btn.previousElementSibling;
            if (input.type === 'password') {
                input.type = 'text';
                btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>'; // Eye Off
            } else {
                input.type = 'password';
                btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>'; // Eye
            }
        });
    });
}

// --- INIT ---

document.addEventListener('DOMContentLoaded', () => {
    VFX.init();
    initFileUpload();
    initReportsTabs();
    initPasswordToggles();
    
    // Global Event Listeners
    document.querySelectorAll('[data-page]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(btn.dataset.page);
        });
    });
    
    document.getElementById('startUploadBtn')?.addEventListener('click', startAnalysis);
    document.getElementById('newAnalysisBtn')?.addEventListener('click', () => navigateTo('upload'));
    document.getElementById('emptyStartBtn')?.addEventListener('click', () => navigateTo('upload'));
    document.getElementById('startAnalysisBtn')?.addEventListener('click', () => navigateTo('upload'));
    
    // Download Report Button
    document.getElementById('downloadReportBtn')?.addEventListener('click', async () => {
        if (!state.currentJobId) {
            showToast('No analysis to download', 'error');
            return;
        }
        try {
            showToast('Generating PDF...', 'info');
            await api.downloadPDF(state.currentJobId);
            showToast('Report downloaded!', 'success');
        } catch (e) {
            showToast('Failed to download PDF: ' + e.message, 'error');
        }
    });

    // Auth Modal Logic
    const authModal = document.getElementById('authModal');
    const tabBtns = document.querySelectorAll('.auth-tab');
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update tabs
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update forms
            const tab = btn.dataset.tab;
            if (tab === 'login') {
                loginForm.style.display = 'block';
                signupForm.style.display = 'none';
            } else {
                loginForm.style.display = 'none';
                signupForm.style.display = 'block';
            }
        });
    });

    // Open Modal Triggers
    document.getElementById('loginBtn')?.addEventListener('click', () => {
        authModal.classList.add('active');
        document.querySelector('[data-tab="login"]').click();
    });

    document.getElementById('signupBtn')?.addEventListener('click', () => {
        authModal.classList.add('active');
        document.querySelector('[data-tab="signup"]').click();
    });

    // Close Modal
    document.getElementById('closeModal')?.addEventListener('click', () => {
        authModal.classList.remove('active');
    });

    // Close on outside click
    authModal.addEventListener('click', (e) => {
        if (e.target === authModal || e.target.classList.contains('modal-overlay')) {
            authModal.classList.remove('active');
        }
    });

    // Dropdown Logic
    const userProfileBtn = document.getElementById('userProfileBtn');
    const userDropdown = document.getElementById('userDropdown');
    const logoutBtn = document.getElementById('logoutBtn');

    userProfileBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        userDropdown.classList.toggle('active');
    });

    document.addEventListener('click', () => {
        userDropdown.classList.remove('active');
    });

    logoutBtn?.addEventListener('click', () => {
        if(confirm('Log out?')) logout();
    });
    
    // Login Logic
    document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const [email, pwd] = e.target.querySelectorAll('input');
        try {
            await api.login(email.value, pwd.value);
            authModal.classList.remove('active');
            updateNavAuth();
            showToast('Welcome back!', 'success');
        } catch(err) {
            showToast(err.message, 'error');
        }
    });

    // Signup Logic
    document.getElementById('signupForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const inputs = e.target.querySelectorAll('input');
        const fullName = inputs[0].value;
        const email = inputs[1].value;
        const pwd = inputs[2].value;
        
        try {
            await api.register(email, pwd, fullName);
            showToast('Account created! Logging in...', 'success');
            
            // Auto Login
            await api.login(email, pwd);
            authModal.classList.remove('active');
            updateNavAuth();
            showToast(`Welcome, ${fullName}!`, 'success');
            
        } catch(err) {
            showToast(err.message, 'error');
        }
    });

    // Initial Auth Check
    if(state.token) {
        api.getUser().then(() => {
            updateNavAuth();
        }).catch(() => {
            // Token invalid
            logout();
        });
    } else {
        updateNavAuth();
    }
});
