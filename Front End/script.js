class CamouflageDetector {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.currentFile = null;
        this.audioEnabled = true;
    }

    initializeElements() {
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.previewArea = document.getElementById('previewArea');
        this.previewImage = document.getElementById('previewImage');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.submitBtn = document.getElementById('submitBtn');
        this.clearBtn = document.getElementById('clearBtn');

        this.speakerBtn = document.getElementById('speakerBtn');
        this.speakerIcon = document.getElementById('speakerIcon');
        
        // Result elements
        this.detectionResults = document.getElementById('detectionResults');
        this.segmentationResults = document.getElementById('segmentationResults');
        this.detectionImage = document.getElementById('detectionImage');
        this.segmentationImage = document.getElementById('segmentationImage');
        this.heatmapResults = document.getElementById('heatmapResults');
        this.heatmapImage = document.getElementById('heatmapImage');
        
        // Preview modal elements
        this.imagePreviewModal = document.getElementById('imagePreviewModal');
        this.closePreviewModal = document.getElementById('closePreviewModal');
        this.previewModalImage = document.getElementById('previewModalImage');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.previewTitle = document.getElementById('previewTitle');
    }

    setupEventListeners() {
        // Drag and drop
        this.dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        this.dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.dropZone.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Buttons
        this.submitBtn.addEventListener('click', this.processImage.bind(this));
        this.clearBtn.addEventListener('click', this.clearImage.bind(this));

        this.speakerBtn.addEventListener('click', this.toggleAudio.bind(this));
        this.closePreviewModal.addEventListener('click', this.closeImagePreview.bind(this));
        this.downloadBtn.addEventListener('click', this.downloadImage.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboard.bind(this));
        
        // Click outside modal to close
        this.imagePreviewModal.addEventListener('click', (e) => {
            if (e.target === this.imagePreviewModal) {
                this.closeImagePreview();
            }
        });
        

    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }

        this.currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewArea.style.display = 'block';
            this.dropZone.querySelector('.drop-content').style.display = 'none';
            this.submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }



    handleKeyboard(e) {
        if (e.key === 'Escape') {
            this.closeImagePreview();
        }
    }
    
    openImagePreview(imageSrc, title, detectionBoxes = null, classifications = null) {
        this.previewModalImage.src = imageSrc;
        this.previewTitle.textContent = title;
        this.imagePreviewModal.style.display = 'flex';
        this.currentPreviewSrc = imageSrc;
        this.currentDetections = detectionBoxes;
        this.currentClassifications = classifications;
        
        // Add detection areas to preview modal if available
        if (detectionBoxes && classifications) {
            setTimeout(() => {
                this.addDetectionAreas(this.imagePreviewModal.querySelector('.preview-image-container'), this.previewModalImage, detectionBoxes, classifications);
            }, 100);
        }
    }
    
    closeImagePreview() {
        this.imagePreviewModal.style.display = 'none';
        this.currentPreviewSrc = null;
        this.currentDetections = null;
        this.currentClassifications = null;
        
        // Clear detection areas from preview
        const container = this.imagePreviewModal.querySelector('.preview-image-container');
        const detectionAreas = container.querySelectorAll('.detection-area');
        detectionAreas.forEach(area => area.remove());
    }
    
    downloadImage() {
        if (this.currentPreviewSrc) {
            const link = document.createElement('a');
            link.href = this.currentPreviewSrc;
            link.download = `camouflage_detection_${Date.now()}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }

    async processImage() {
        if (!this.currentFile) return;

        this.showLoading(true);
        this.submitBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Upload failed: ${response.status} - ${errorText}`);
            }

            const results = await response.json();
            this.displayResults(results);

        } catch (error) {
            console.error('Full error:', error);
            alert('Error processing image: ' + error.message);
        } finally {
            this.showLoading(false);
            this.submitBtn.disabled = false;
        }
    }

    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    displayResults(results) {
        // Debug: Log the results to see the data structure
        console.log('Results received:', results);
        
        // Clear previous results
        this.clearResults();

        // 1. Detection Results
        if (results.results && results.results.length > 0) {
            const detectionHTML = results.results.map((detection, index) => {
                return `
                    <div class="object-item">
                        <span>🎯 ${detection.object_name}</span>
                        <span class="confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            }).join('');
            
            this.detectionResults.innerHTML = detectionHTML;
        } else {
            this.detectionResults.innerHTML = '<p class="no-results">No camouflaged objects detected</p>';
        }

        // 2. Segmentation Results
        if (results.results && results.results.length > 0) {
            this.segmentationResults.innerHTML = `<p>🎨 ${results.results.length} objects segmented with pixel-perfect masks</p>`;
        } else {
            this.segmentationResults.innerHTML = '<p>🎨 No objects found to segment</p>';
        }
        
        // 3. Heatmap Results
        if (results.results && results.results.length > 0) {
            this.heatmapResults.innerHTML = `<p>🔥 Probability heatmap showing ${results.results.length} detection regions</p>`;
        } else {
            this.heatmapResults.innerHTML = '<p>🔥 No probability regions detected</p>';
        }

        // Display result images
        if (results.images) {
            // Detection image with bounding boxes
            if (results.images.detection) {
                const detImg = document.createElement('img');
                detImg.src = `data:image/jpeg;base64,${results.images.detection}`;
                detImg.alt = 'Detection Result';
                detImg.title = 'Click to view fullscreen';
                detImg.addEventListener('click', () => {
                    this.openImagePreview(detImg.src, 'Object Detection with Bounding Boxes');
                });
                this.detectionImage.innerHTML = '';
                this.detectionImage.appendChild(detImg);
            }
            
            // Segmentation image with masks
            if (results.images.segmentation) {
                const segImg = document.createElement('img');
                segImg.src = `data:image/jpeg;base64,${results.images.segmentation}`;
                segImg.alt = 'Segmentation Result';
                segImg.title = 'Click to view fullscreen';
                segImg.addEventListener('click', () => {
                    this.openImagePreview(segImg.src, 'Pixel-Perfect Segmentation Masks');
                });
                this.segmentationImage.innerHTML = '';
                this.segmentationImage.appendChild(segImg);
            }
            
            // Heatmap image
            if (results.images.heatmap) {
                const heatImg = document.createElement('img');
                heatImg.src = `data:image/jpeg;base64,${results.images.heatmap}`;
                heatImg.alt = 'Probability Heatmap';
                heatImg.title = 'Click to view fullscreen';
                heatImg.addEventListener('click', () => {
                    this.openImagePreview(heatImg.src, 'Probability Heatmap');
                });
                this.heatmapImage.innerHTML = '';
                this.heatmapImage.appendChild(heatImg);
            }
        }
    }
    
    addDetectionAreas(container, img, detectionBoxes, classifications) {
        console.log('Adding detection areas:', detectionBoxes, classifications);
        
        const addAreas = () => {
            // Clear existing detection areas
            const existingAreas = container.querySelectorAll('.detection-area');
            existingAreas.forEach(area => area.remove());
            
            if (!detectionBoxes || !classifications) return;
            
            // Get image position relative to container
            const imgRect = img.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            // Calculate scale factors
            const scaleX = img.clientWidth / img.naturalWidth;
            const scaleY = img.clientHeight / img.naturalHeight;
            
            // Calculate image offset within container
            const offsetX = imgRect.left - containerRect.left;
            const offsetY = imgRect.top - containerRect.top;
            
            detectionBoxes.forEach((box, index) => {
                if (classifications[index]) {
                    const detectionArea = document.createElement('div');
                    detectionArea.className = 'detection-area';
                    detectionArea.style.cssText = `
                        position: absolute;
                        left: ${offsetX + (box.x1 * scaleX)}px;
                        top: ${offsetY + (box.y1 * scaleY)}px;
                        width: ${(box.x2 - box.x1) * scaleX}px;
                        height: ${(box.y2 - box.y1) * scaleY}px;
                        background: rgba(255, 0, 0, 0.1);
                        border: 2px solid rgba(255, 0, 0, 0.3);
                        cursor: pointer;
                        z-index: 100;
                    `;
                    
                    // Add hover event for this specific detection
                    detectionArea.addEventListener('mouseenter', () => {
                        console.log('Hovering over detection:', classifications[index]);
                        const objectName = classifications[index].class.replace('camouflaged_', '').replace('_object', '');
                        this.announceObjects(objectName);
                    });
                    
                    container.appendChild(detectionArea);
                }
            });
        };
        
        // Wait for image to load
        if (img.complete) {
            setTimeout(addAreas, 100);
        } else {
            img.onload = () => setTimeout(addAreas, 100);
        }
    }

    announceObjects(objectNames) {
        // Text-to-speech announcement (only if audio enabled)
        if (this.audioEnabled && 'speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(`Detected objects: ${objectNames}`);
            utterance.rate = 0.8;
            utterance.pitch = 1;
            speechSynthesis.speak(utterance);
        }
        
        // Visual notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4299e1;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        notification.innerHTML = `${this.audioEnabled ? '🔊' : '🔇'} Detected: ${objectNames}`;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    clearImage() {
        this.currentFile = null;
        this.previewArea.style.display = 'none';
        this.dropZone.querySelector('.drop-content').style.display = 'flex';
        this.submitBtn.disabled = true;
        this.fileInput.value = '';
        
        // Clear results
        this.detectionResults.innerHTML = '<p class="no-results">Upload an image to detect camouflaged objects</p>';
        this.segmentationResults.innerHTML = '<p class="no-results">Upload an image to see processing info</p>';
        this.heatmapResults.innerHTML = '<p class="no-results">Upload an image to see session info</p>';
        this.detectionImage.innerHTML = '';
        this.segmentationImage.innerHTML = '';
        this.heatmapImage.innerHTML = '';
    }

    clearResults() {
        this.detectionResults.innerHTML = '<p class="no-results">Processing...</p>';
        this.segmentationResults.innerHTML = '<p class="no-results">Processing...</p>';
        this.heatmapResults.innerHTML = '<p class="no-results">Processing...</p>';
        this.detectionImage.innerHTML = '';
        this.segmentationImage.innerHTML = '';
        this.heatmapImage.innerHTML = '';
        this.closeImagePreview();
    }

    toggleAudio() {
        this.audioEnabled = !this.audioEnabled;
        this.speakerIcon.textContent = this.audioEnabled ? '🔊' : '🔇';
        this.speakerBtn.title = this.audioEnabled ? 'Mute audio announcements' : 'Enable audio announcements';
        
        // Stop any current speech
        if ('speechSynthesis' in window) {
            speechSynthesis.cancel();
        }
    }
}

// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new CamouflageDetector();
});