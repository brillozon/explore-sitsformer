Model Deployment and Production
================================

This tutorial covers deploying SITS-Former models for production use, including model optimization, containerization, API development, batch processing, and cloud deployment strategies.

Overview
--------

Deploying satellite image time series models involves several considerations:

1. **Model Optimization** - Quantization, pruning, and ONNX conversion
2. **Containerization** - Docker packaging for consistent deployments
3. **API Development** - REST APIs for model serving
4. **Batch Processing** - Large-scale inference pipelines
5. **Cloud Deployment** - AWS, Azure, and GCP deployment options
6. **Monitoring** - Performance tracking and model drift detection

Prerequisites
-------------

Before deploying your model, ensure you have:

- A trained and validated SITS-Former model
- Understanding of your deployment requirements (latency, throughput, cost)
- Access to deployment infrastructure (cloud or on-premises)

Model Optimization for Production
---------------------------------

Model Quantization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.quantization as quantization
    from sitsformer.models import SITSFormer
    
    def quantize_model(model, calibration_loader, device='cpu'):
        """Quantize model for faster inference and smaller size."""
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        # Prepare model for quantization
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        print("Calibrating quantized model...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= 50:  # Use limited samples for calibration
                    break
                    
                sequences = batch['sequence'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
                
                # Forward pass for calibration
                if masks is not None:
                    _ = model(sequences, masks)
                else:
                    _ = model(sequences)
        
        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)
        
        print("Model quantization completed")
        return quantized_model
    
    def compare_model_sizes(original_model, quantized_model):
        """Compare original and quantized model sizes."""
        
        # Save models temporarily to measure size
        torch.save(original_model.state_dict(), 'temp_original.pth')
        torch.save(quantized_model.state_dict(), 'temp_quantized.pth')
        
        import os
        original_size = os.path.getsize('temp_original.pth') / (1024 * 1024)  # MB
        quantized_size = os.path.getsize('temp_quantized.pth') / (1024 * 1024)  # MB
        
        # Cleanup
        os.remove('temp_original.pth')
        os.remove('temp_quantized.pth')
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        return original_size, quantized_size

ONNX Conversion
~~~~~~~~~~~~~~~

.. code-block:: python

    import torch.onnx
    import onnxruntime as ort
    import numpy as np
    
    def convert_to_onnx(model, sample_input, onnx_path, dynamic_axes=None):
        """Convert PyTorch model to ONNX format."""
        
        model.eval()
        
        # Define input/output names
        input_names = ['sequences']
        output_names = ['predictions']
        
        if isinstance(sample_input, tuple):
            input_names = ['sequences', 'masks']
            sample_sequences, sample_masks = sample_input
        else:
            sample_sequences = sample_input
            sample_masks = None
        
        # Dynamic axes for variable batch size and sequence length
        if dynamic_axes is None:
            dynamic_axes = {
                'sequences': {0: 'batch_size', 1: 'sequence_length'},
                'predictions': {0: 'batch_size'}
            }
            if sample_masks is not None:
                dynamic_axes['masks'] = {0: 'batch_size', 1: 'sequence_length'}
        
        # Export to ONNX
        with torch.no_grad():
            if sample_masks is not None:
                torch.onnx.export(
                    model,
                    (sample_sequences, sample_masks),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
                )
            else:
                torch.onnx.export(
                    model,
                    sample_sequences,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
                )
        
        print(f"Model exported to ONNX: {onnx_path}")
        return onnx_path
    
    def verify_onnx_model(onnx_path, pytorch_model, sample_input):
        """Verify ONNX model produces same outputs as PyTorch model."""
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Prepare inputs
        if isinstance(sample_input, tuple):
            sequences, masks = sample_input
            ort_inputs = {
                'sequences': sequences.numpy(),
                'masks': masks.numpy()
            }
            pytorch_input = (sequences, masks)
        else:
            ort_inputs = {'sequences': sample_input.numpy()}
            pytorch_input = sample_input
        
        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            if isinstance(pytorch_input, tuple):
                pytorch_output = pytorch_model(*pytorch_input)
            else:
                pytorch_output = pytorch_model(pytorch_input)
        
        # Get ONNX output
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outputs[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output.numpy() - onnx_output).max()
        print(f"Max difference between PyTorch and ONNX outputs: {diff}")
        
        if diff < 1e-5:
            print("✓ ONNX conversion successful - outputs match")
            return True
        else:
            print("✗ ONNX conversion issue - outputs differ significantly")
            return False

Containerization with Docker
-----------------------------

Dockerfile Creation
~~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

    # Dockerfile for SITS-Former deployment
    FROM python:3.9-slim
    
    # Set working directory
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \\
        gcc \\
        g++ \\
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements first for better caching
    COPY requirements.txt .
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy model and application code
    COPY models/ models/
    COPY src/ src/
    COPY app.py .
    
    # Set environment variables
    ENV PYTHONPATH=/app/src
    ENV MODEL_PATH=/app/models/best_model.pth
    ENV DEVICE=cpu
    
    # Expose port
    EXPOSE 8000
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
        CMD curl -f http://localhost:8000/health || exit 1
    
    # Run application
    CMD ["python", "app.py"]

Multi-stage Build for Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

    # Multi-stage Dockerfile for optimized production deployment
    
    # Stage 1: Build environment
    FROM python:3.9 as builder
    
    WORKDIR /app
    
    # Install build dependencies
    RUN apt-get update && apt-get install -y \\
        gcc \\
        g++ \\
        git
    
    # Copy and install Python dependencies
    COPY requirements.txt .
    RUN pip install --user --no-cache-dir -r requirements.txt
    
    # Copy and build application
    COPY . .
    RUN pip install --user .
    
    # Stage 2: Production environment
    FROM python:3.9-slim as production
    
    WORKDIR /app
    
    # Install runtime dependencies only
    RUN apt-get update && apt-get install -y \\
        curl \\
        && rm -rf /var/lib/apt/lists/*
    
    # Copy Python packages from builder
    COPY --from=builder /root/.local /root/.local
    
    # Copy application files
    COPY --from=builder /app/models /app/models
    COPY --from=builder /app/src /app/src
    COPY app.py .
    
    # Set environment
    ENV PATH=/root/.local/bin:$PATH
    ENV PYTHONPATH=/app/src
    
    # Create non-root user
    RUN useradd --create-home --shell /bin/bash appuser
    USER appuser
    
    EXPOSE 8000
    
    CMD ["python", "app.py"]

Docker Compose for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'
    
    services:
      sits-former-api:
        build: .
        ports:
          - "8000:8000"
        environment:
          - MODEL_PATH=/app/models/best_model.pth
          - DEVICE=cpu
          - LOG_LEVEL=INFO
        volumes:
          - ./models:/app/models
          - ./logs:/app/logs
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 10s
          retries: 3
      
      nginx:
        image: nginx:alpine
        ports:
          - "80:80"
        volumes:
          - ./nginx.conf:/etc/nginx/nginx.conf
        depends_on:
          - sits-former-api
        restart: unless-stopped
      
      redis:
        image: redis:alpine
        ports:
          - "6379:6379"
        restart: unless-stopped

REST API Development
--------------------

FastAPI Implementation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # app.py - FastAPI application for SITS-Former serving
    
    import os
    import asyncio
    import numpy as np
    import torch
    from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    import logging
    from datetime import datetime
    import redis
    import json
    
    from sitsformer.models import SITSFormer
    from sitsformer.inference import SITSFormerPredictor
    from sitsformer.preprocessing import preprocess_satellite_data
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize FastAPI app
    app = FastAPI(
        title="SITS-Former API",
        description="Satellite Image Time Series Classification API",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global variables
    model = None
    predictor = None
    redis_client = None
    class_names = ['Water', 'Forest', 'Grassland', 'Cropland', 'Urban', 
                   'Bare Soil', 'Snow/Ice', 'Cloud', 'Shadow', 'Wetland']
    
    # Pydantic models for API
    class PredictionRequest(BaseModel):
        sequences: List[List[List[List[float]]]]  # [T, C, H, W] format
        masks: Optional[List[List[bool]]] = None  # [T] format
        metadata: Optional[Dict[str, Any]] = {}
    
    class PredictionResponse(BaseModel):
        prediction: int
        class_name: str
        confidence: float
        probabilities: List[float]
        processing_time: float
        metadata: Dict[str, Any]
    
    class BatchPredictionRequest(BaseModel):
        batch_id: str
        sequences_list: List[List[List[List[List[float]]]]]  # Batch of sequences
        masks_list: Optional[List[List[List[bool]]]] = None
        metadata_list: Optional[List[Dict[str, Any]]] = []
    
    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        timestamp: str
        version: str
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize model and services on startup."""
        global model, predictor, redis_client
        
        try:
            # Load model
            model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
            device = os.getenv('DEVICE', 'cpu')
            
            logger.info(f"Loading model from {model_path}")
            predictor = SITSFormerPredictor(model_path, device=device)
            
            # Initialize Redis for caching (optional)
            try:
                redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
                redis_client.ping()
                logger.info("Redis connection established")
            except:
                logger.warning("Redis not available - caching disabled")
                redis_client = None
            
            logger.info("SITS-Former API started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API: {str(e)}")
            raise
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if predictor is not None else "unhealthy",
            model_loaded=predictor is not None,
            timestamp=datetime.now().isoformat(),
            version="1.0.0"
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_single(request: PredictionRequest):
        """Single prediction endpoint."""
        
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            start_time = datetime.now()
            
            # Convert input to tensors
            sequences = torch.tensor(request.sequences, dtype=torch.float32)
            masks = None
            if request.masks:
                masks = torch.tensor(request.masks, dtype=torch.bool)
            
            # Generate prediction
            result = predictor.predict(sequences, masks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Cache result if Redis available
            if redis_client:
                cache_key = f"prediction:{hash(str(request.sequences))}"
                cache_data = {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                    "timestamp": start_time.isoformat()
                }
                redis_client.setex(cache_key, 3600, json.dumps(cache_data))  # 1 hour TTL
            
            return PredictionResponse(
                prediction=result["prediction"],
                class_name=class_names[result["prediction"]],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                processing_time=processing_time,
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post("/predict/batch")
    async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
        """Batch prediction endpoint with async processing."""
        
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Start background processing
        background_tasks.add_task(process_batch, request)
        
        return {
            "batch_id": request.batch_id,
            "status": "processing",
            "message": "Batch prediction started",
            "estimated_completion": "Check /batch/{batch_id}/status"
        }
    
    async def process_batch(request: BatchPredictionRequest):
        """Process batch prediction in background."""
        
        try:
            results = []
            
            for i, sequences in enumerate(request.sequences_list):
                masks = None
                if request.masks_list and i < len(request.masks_list):
                    masks = torch.tensor(request.masks_list[i], dtype=torch.bool)
                
                sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
                result = predictor.predict(sequences_tensor, masks)
                results.append(result)
            
            # Store results in Redis
            if redis_client:
                batch_result = {
                    "batch_id": request.batch_id,
                    "status": "completed",
                    "results": results,
                    "completed_at": datetime.now().isoformat()
                }
                redis_client.setex(f"batch:{request.batch_id}", 86400, json.dumps(batch_result))
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            if redis_client:
                error_result = {
                    "batch_id": request.batch_id,
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                }
                redis_client.setex(f"batch:{request.batch_id}", 86400, json.dumps(error_result))
    
    @app.get("/batch/{batch_id}/status")
    async def get_batch_status(batch_id: str):
        """Get batch processing status."""
        
        if not redis_client:
            raise HTTPException(status_code=503, detail="Batch tracking not available")
        
        batch_data = redis_client.get(f"batch:{batch_id}")
        
        if not batch_data:
            raise HTTPException(status_code=404, detail="Batch ID not found")
        
        return json.loads(batch_data)
    
    @app.post("/predict/file")
    async def predict_from_file(file: UploadFile = File(...)):
        """Predict from uploaded file (e.g., GeoTIFF)."""
        
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            # Process the file
            sequences, masks = preprocess_satellite_data(tmp_file_path)
            result = predictor.predict(sequences, masks)
            
            # Cleanup
            os.unlink(tmp_file_path)
            
            return {
                "filename": file.filename,
                "prediction": result["prediction"],
                "class_name": class_names[result["prediction"]],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            }
            
        except Exception as e:
            logger.error(f"File prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
    
    @app.get("/models/info")
    async def get_model_info():
        """Get model information."""
        
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "model_type": "SITS-Former",
            "classes": class_names,
            "num_classes": len(class_names),
            "device": str(predictor.device),
            "parameters": predictor.get_model_info()
        }
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            workers=1,  # Use 1 worker to avoid model loading multiple times
            log_level="info"
        )

Inference Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # sitsformer/inference.py - Optimized inference class
    
    import torch
    import torch.nn.functional as F
    import time
    from typing import Optional, Dict, Any, List
    import numpy as np
    
    class SITSFormerPredictor:
        """Optimized SITS-Former predictor for production deployment."""
        
        def __init__(self, model_path: str, device: str = 'auto', 
                     use_half_precision: bool = False, batch_size: int = 1):
            """
            Initialize predictor.
            
            Args:
                model_path: Path to model checkpoint
                device: Device to use ('auto', 'cpu', 'cuda')
                use_half_precision: Use FP16 for inference
                batch_size: Maximum batch size for batch inference
            """
            
            self.device = self._setup_device(device)
            self.use_half_precision = use_half_precision and self.device.type == 'cuda'
            self.max_batch_size = batch_size
            
            # Load model
            self.model = self._load_model(model_path)
            self.model.eval()
            
            # Warm up model
            self._warmup()
            
        def _setup_device(self, device: str) -> torch.device:
            """Setup compute device."""
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
            return torch.device(device)
        
        def _load_model(self, model_path: str):
            """Load model from checkpoint."""
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Get model configuration
            model_config = checkpoint.get('model_config', {})
            
            # Create model
            from sitsformer.models import SITSFormer
            model = SITSFormer(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            
            if self.use_half_precision:
                model = model.half()
            
            return model
        
        def _warmup(self, num_warmup: int = 3):
            """Warm up model for consistent timing."""
            dummy_input = torch.randn(1, 10, 13, 64, 64).to(self.device)
            if self.use_half_precision:
                dummy_input = dummy_input.half()
            
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = self.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        def predict(self, sequences: torch.Tensor, 
                   masks: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            """
            Single prediction.
            
            Args:
                sequences: Input sequences [T, C, H, W] or [B, T, C, H, W]
                masks: Optional masks [T] or [B, T]
                
            Returns:
                Dictionary with prediction results
            """
            
            # Ensure batch dimension
            if sequences.dim() == 4:
                sequences = sequences.unsqueeze(0)
                if masks is not None:
                    masks = masks.unsqueeze(0)
            
            # Move to device and convert precision
            sequences = sequences.to(self.device)
            if self.use_half_precision:
                sequences = sequences.half()
            
            if masks is not None:
                masks = masks.to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                if masks is not None:
                    outputs = self.model(sequences, masks)
                else:
                    outputs = self.model(sequences)
                
                # Get probabilities and prediction
                probabilities = F.softmax(outputs, dim=1)
                prediction = outputs.argmax(dim=1)
                confidence = probabilities.max(dim=1)[0]
            
            inference_time = time.time() - start_time
            
            return {
                "prediction": prediction.item(),
                "confidence": confidence.item(),
                "probabilities": probabilities.squeeze().cpu().numpy().tolist(),
                "inference_time": inference_time
            }
        
        def predict_batch(self, sequences_list: List[torch.Tensor],
                         masks_list: Optional[List[torch.Tensor]] = None) -> List[Dict[str, Any]]:
            """
            Batch prediction with automatic batching.
            
            Args:
                sequences_list: List of sequences
                masks_list: Optional list of masks
                
            Returns:
                List of prediction results
            """
            
            results = []
            
            for i in range(0, len(sequences_list), self.max_batch_size):
                batch_end = min(i + self.max_batch_size, len(sequences_list))
                batch_sequences = sequences_list[i:batch_end]
                batch_masks = masks_list[i:batch_end] if masks_list else None
                
                # Stack batch
                stacked_sequences = torch.stack(batch_sequences)
                stacked_masks = None
                if batch_masks:
                    stacked_masks = torch.stack(batch_masks)
                
                # Predict batch
                batch_result = self._predict_batch_internal(stacked_sequences, stacked_masks)
                results.extend(batch_result)
            
            return results
        
        def _predict_batch_internal(self, sequences: torch.Tensor,
                                  masks: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
            """Internal batch prediction."""
            
            sequences = sequences.to(self.device)
            if self.use_half_precision:
                sequences = sequences.half()
            
            if masks is not None:
                masks = masks.to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                if masks is not None:
                    outputs = self.model(sequences, masks)
                else:
                    outputs = self.model(sequences)
                
                probabilities = F.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                confidences = probabilities.max(dim=1)[0]
            
            inference_time = time.time() - start_time
            
            # Convert to list of results
            results = []
            for i in range(len(predictions)):
                results.append({
                    "prediction": predictions[i].item(),
                    "confidence": confidences[i].item(),
                    "probabilities": probabilities[i].cpu().numpy().tolist(),
                    "inference_time": inference_time / len(predictions)
                })
            
            return results
        
        def get_model_info(self) -> Dict[str, Any]:
            """Get model information."""
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "device": str(self.device),
                "half_precision": self.use_half_precision,
                "max_batch_size": self.max_batch_size
            }

Cloud Deployment Strategies
---------------------------

AWS Deployment
~~~~~~~~~~~~~~

.. code-block:: python

    # deploy_aws.py - AWS deployment script
    
    import boto3
    import json
    import zipfile
    import os
    from datetime import datetime
    
    class AWSDeployer:
        """Deploy SITS-Former to AWS using Lambda + API Gateway or ECS."""
        
        def __init__(self, aws_region='us-east-1'):
            self.region = aws_region
            self.lambda_client = boto3.client('lambda', region_name=aws_region)
            self.apigateway_client = boto3.client('apigateway', region_name=aws_region)
            self.ecs_client = boto3.client('ecs', region_name=aws_region)
            self.ecr_client = boto3.client('ecr', region_name=aws_region)
        
        def deploy_lambda(self, model_path, function_name='sits-former-predictor'):
            """Deploy as AWS Lambda function."""
            
            # Create deployment package
            zip_path = self._create_lambda_package(model_path)
            
            # Upload to Lambda
            with open(zip_path, 'rb') as zip_file:
                zip_data = zip_file.read()
            
            try:
                # Try to update existing function
                response = self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_data
                )
                print(f"Updated existing Lambda function: {function_name}")
                
            except self.lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                response = self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role='arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role',
                    Handler='lambda_handler.handler',
                    Code={'ZipFile': zip_data},
                    Timeout=300,  # 5 minutes
                    MemorySize=3008,  # Maximum memory for models
                    Environment={
                        'Variables': {
                            'MODEL_PATH': '/tmp/model.pth'
                        }
                    }
                )
                print(f"Created new Lambda function: {function_name}")
            
            return response['FunctionArn']
        
        def deploy_ecs(self, image_uri, cluster_name='sits-former-cluster',
                      service_name='sits-former-service'):
            """Deploy as ECS service."""
            
            # Task definition
            task_definition = {
                'family': 'sits-former-task',
                'networkMode': 'awsvpc',
                'requiresCompatibilities': ['FARGATE'],
                'cpu': '2048',  # 2 vCPU
                'memory': '4096',  # 4 GB
                'executionRoleArn': 'arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole',
                'containerDefinitions': [
                    {
                        'name': 'sits-former-container',
                        'image': image_uri,
                        'portMappings': [
                            {
                                'containerPort': 8000,
                                'protocol': 'tcp'
                            }
                        ],
                        'essential': True,
                        'logConfiguration': {
                            'logDriver': 'awslogs',
                            'options': {
                                'awslogs-group': '/ecs/sits-former',
                                'awslogs-region': self.region,
                                'awslogs-stream-prefix': 'ecs'
                            }
                        },
                        'environment': [
                            {'name': 'DEVICE', 'value': 'cpu'},
                            {'name': 'LOG_LEVEL', 'value': 'INFO'}
                        ]
                    }
                ]
            }
            
            # Register task definition
            task_response = self.ecs_client.register_task_definition(**task_definition)
            task_def_arn = task_response['taskDefinition']['taskDefinitionArn']
            
            # Create or update service
            service_config = {
                'serviceName': service_name,
                'cluster': cluster_name,
                'taskDefinition': task_def_arn,
                'desiredCount': 2,
                'launchType': 'FARGATE',
                'networkConfiguration': {
                    'awsvpcConfiguration': {
                        'subnets': ['subnet-12345', 'subnet-67890'],  # Your subnet IDs
                        'securityGroups': ['sg-12345'],  # Your security group ID
                        'assignPublicIp': 'ENABLED'
                    }
                }
            }
            
            try:
                # Update existing service
                self.ecs_client.update_service(**service_config)
                print(f"Updated ECS service: {service_name}")
            except:
                # Create new service
                self.ecs_client.create_service(**service_config)
                print(f"Created ECS service: {service_name}")
            
            return task_def_arn

Azure Deployment
~~~~~~~~~~~~~~~~

.. code-block:: python

    # deploy_azure.py - Azure deployment script
    
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient
    from azure.mgmt.web import WebSiteManagementClient
    
    class AzureDeployer:
        """Deploy SITS-Former to Azure Container Instances or App Service."""
        
        def __init__(self, subscription_id, resource_group):
            self.subscription_id = subscription_id
            self.resource_group = resource_group
            self.credential = DefaultAzureCredential()
        
        def deploy_container_instance(self, image_name, container_name='sits-former-ci'):
            """Deploy to Azure Container Instances."""
            
            aci_client = ContainerInstanceManagementClient(
                self.credential, self.subscription_id
            )
            
            container_group = {
                'location': 'East US',
                'containers': [
                    {
                        'name': container_name,
                        'image': image_name,
                        'resources': {
                            'requests': {
                                'cpu': 2.0,
                                'memory_in_gb': 4.0
                            }
                        },
                        'ports': [
                            {
                                'protocol': 'TCP',
                                'port': 8000
                            }
                        ],
                        'environment_variables': [
                            {
                                'name': 'DEVICE',
                                'value': 'cpu'
                            }
                        ]
                    }
                ],
                'os_type': 'Linux',
                'ip_address': {
                    'type': 'Public',
                    'ports': [
                        {
                            'protocol': 'TCP',
                            'port': 8000
                        }
                    ]
                },
                'restart_policy': 'Always'
            }
            
            # Create container group
            operation = aci_client.container_groups.begin_create_or_update(
                self.resource_group,
                container_name,
                container_group
            )
            
            result = operation.result()
            print(f"Container instance created: {result.name}")
            return result
        
        def deploy_app_service(self, image_name, app_name='sits-former-app'):
            """Deploy to Azure App Service."""
            
            web_client = WebSiteManagementClient(
                self.credential, self.subscription_id
            )
            
            # App Service configuration
            site_config = {
                'location': 'East US',
                'server_farm_id': f'/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Web/serverfarms/sits-former-plan',
                'site_config': {
                    'linux_fx_version': f'DOCKER|{image_name}',
                    'always_on': True,
                    'app_settings': [
                        {
                            'name': 'WEBSITES_ENABLE_APP_SERVICE_STORAGE',
                            'value': 'false'
                        },
                        {
                            'name': 'DEVICE',
                            'value': 'cpu'
                        }
                    ]
                }
            }
            
            # Create or update app service
            operation = web_client.web_apps.begin_create_or_update(
                self.resource_group,
                app_name,
                site_config
            )
            
            result = operation.result()
            print(f"App Service created: {result.name}")
            return result

Google Cloud Platform Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # gcp-deployment.yaml - Google Cloud Run deployment
    
    apiVersion: serving.knative.dev/v1
    kind: Service
    metadata:
      name: sits-former-service
      annotations:
        run.googleapis.com/ingress: all
    spec:
      template:
        metadata:
          annotations:
            autoscaling.knative.dev/maxScale: "10"
            run.googleapis.com/cpu-throttling: "false"
        spec:
          containerConcurrency: 4
          containers:
          - image: gcr.io/YOUR_PROJECT/sits-former:latest
            ports:
            - containerPort: 8000
            env:
            - name: DEVICE
              value: "cpu"
            - name: LOG_LEVEL
              value: "INFO"
            resources:
              limits:
                cpu: "2"
                memory: "4Gi"
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 30

Batch Processing Pipeline
-------------------------

Kubernetes Batch Job
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # batch-job.yaml - Kubernetes batch processing job
    
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: sits-former-batch-processing
    spec:
      parallelism: 4
      completions: 1
      backoffLimit: 3
      template:
        spec:
          containers:
          - name: sits-former-batch
            image: sits-former:latest
            command: ["python", "batch_processor.py"]
            args: ["--input-path", "/data/input", "--output-path", "/data/output"]
            env:
            - name: DEVICE
              value: "cpu"
            - name: BATCH_SIZE
              value: "32"
            volumeMounts:
            - name: data-volume
              mountPath: /data
            resources:
              requests:
                cpu: "1"
                memory: "2Gi"
              limits:
                cpu: "2"
                memory: "4Gi"
          volumes:
          - name: data-volume
            persistentVolumeClaim:
              claimName: sits-data-pvc
          restartPolicy: Never

Apache Airflow Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # airflow_pipeline.py - Airflow DAG for batch processing
    
    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    
    default_args = {
        'owner': 'sits-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    }
    
    dag = DAG(
        'sits_former_batch_pipeline',
        default_args=default_args,
        description='SITS-Former batch processing pipeline',
        schedule_interval='@daily',
        catchup=False
    )
    
    # Data preprocessing task
    preprocess_data = KubernetesPodOperator(
        task_id='preprocess_satellite_data',
        name='preprocess-data',
        namespace='sits-processing',
        image='sits-former:latest',
        cmds=['python', 'preprocess.py'],
        arguments=['--date', '{{ ds }}'],
        dag=dag
    )
    
    # Model inference task
    run_inference = KubernetesPodOperator(
        task_id='run_sits_inference',
        name='sits-inference',
        namespace='sits-processing',
        image='sits-former:latest',
        cmds=['python', 'batch_inference.py'],
        arguments=['--input-date', '{{ ds }}'],
        dag=dag
    )
    
    # Post-processing task
    postprocess_results = KubernetesPodOperator(
        task_id='postprocess_results',
        name='postprocess-results',
        namespace='sits-processing',
        image='sits-former:latest',
        cmds=['python', 'postprocess.py'],
        arguments=['--date', '{{ ds }}'],
        dag=dag
    )
    
    # Set task dependencies
    preprocess_data >> run_inference >> postprocess_results

Monitoring and Logging
-----------------------

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # monitoring.py - Prometheus metrics for model monitoring
    
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import time
    import functools
    
    # Define metrics
    PREDICTION_COUNTER = Counter('sits_predictions_total', 'Total predictions made', ['model_version', 'class'])
    PREDICTION_LATENCY = Histogram('sits_prediction_duration_seconds', 'Prediction latency')
    MODEL_CONFIDENCE = Histogram('sits_prediction_confidence', 'Prediction confidence scores')
    ACTIVE_REQUESTS = Gauge('sits_active_requests', 'Number of active requests')
    ERROR_COUNTER = Counter('sits_errors_total', 'Total errors', ['error_type'])
    
    def monitor_predictions(func):
        """Decorator to monitor prediction metrics."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record metrics
                duration = time.time() - start_time
                PREDICTION_LATENCY.observe(duration)
                MODEL_CONFIDENCE.observe(result.get('confidence', 0))
                PREDICTION_COUNTER.labels(
                    model_version='1.0', 
                    class=result.get('class_name', 'unknown')
                ).inc()
                
                return result
                
            except Exception as e:
                ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()
        
        return wrapper
    
    # Start Prometheus metrics server
    start_http_server(8001)

Model Drift Detection
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # drift_detection.py - Model drift monitoring
    
    import numpy as np
    from scipy import stats
    import logging
    from typing import List, Dict, Any
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    
    @dataclass
    class DriftAlert:
        timestamp: datetime
        metric: str
        current_value: float
        baseline_value: float
        drift_score: float
        severity: str
    
    class ModelDriftDetector:
        """Detect model performance drift over time."""
        
        def __init__(self, baseline_window=7, alert_threshold=0.1):
            self.baseline_window = baseline_window
            self.alert_threshold = alert_threshold
            self.prediction_history = []
            self.performance_history = []
            self.logger = logging.getLogger(__name__)
        
        def log_prediction(self, prediction: Dict[str, Any]):
            """Log prediction for drift analysis."""
            prediction['timestamp'] = datetime.now()
            self.prediction_history.append(prediction)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(days=30)
            self.prediction_history = [
                p for p in self.prediction_history 
                if p['timestamp'] > cutoff_time
            ]
        
        def check_distribution_drift(self) -> List[DriftAlert]:
            """Check for drift in prediction distribution."""
            alerts = []
            
            if len(self.prediction_history) < 100:
                return alerts
            
            # Get recent vs baseline predictions
            recent_time = datetime.now() - timedelta(days=1)
            baseline_time = datetime.now() - timedelta(days=self.baseline_window)
            
            recent_preds = [
                p['prediction'] for p in self.prediction_history
                if p['timestamp'] > recent_time
            ]
            
            baseline_preds = [
                p['prediction'] for p in self.prediction_history
                if baseline_time <= p['timestamp'] <= recent_time
            ]
            
            if len(recent_preds) < 10 or len(baseline_preds) < 10:
                return alerts
            
            # Statistical tests for drift
            # 1. KS test for distribution change
            ks_stat, ks_pvalue = stats.ks_2samp(recent_preds, baseline_preds)
            
            if ks_pvalue < 0.05:  # Significant distribution change
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    metric='prediction_distribution',
                    current_value=ks_stat,
                    baseline_value=0.0,
                    drift_score=ks_stat,
                    severity='high' if ks_stat > 0.2 else 'medium'
                ))
            
            # 2. Check confidence drift
            recent_conf = [
                p['confidence'] for p in self.prediction_history
                if p['timestamp'] > recent_time
            ]
            
            baseline_conf = [
                p['confidence'] for p in self.prediction_history
                if baseline_time <= p['timestamp'] <= recent_time
            ]
            
            if recent_conf and baseline_conf:
                conf_drift = abs(np.mean(recent_conf) - np.mean(baseline_conf))
                
                if conf_drift > self.alert_threshold:
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        metric='confidence_drift',
                        current_value=np.mean(recent_conf),
                        baseline_value=np.mean(baseline_conf),
                        drift_score=conf_drift,
                        severity='high' if conf_drift > 0.2 else 'medium'
                    ))
            
            return alerts
        
        def generate_drift_report(self) -> Dict[str, Any]:
            """Generate comprehensive drift report."""
            alerts = self.check_distribution_drift()
            
            # Summary statistics
            recent_time = datetime.now() - timedelta(days=1)
            recent_preds = [
                p for p in self.prediction_history
                if p['timestamp'] > recent_time
            ]
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_predictions_24h': len(recent_preds),
                'alerts': [
                    {
                        'metric': alert.metric,
                        'severity': alert.severity,
                        'drift_score': alert.drift_score,
                        'timestamp': alert.timestamp.isoformat()
                    } for alert in alerts
                ],
                'class_distribution': self._compute_class_distribution(recent_preds),
                'confidence_stats': self._compute_confidence_stats(recent_preds)
            }
            
            return report

Deployment Checklist
--------------------

Pre-deployment Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Model Validation**
   - Accuracy meets requirements
   - Latency within acceptable limits
   - Memory usage optimized
   - Error handling tested

2. **Infrastructure Testing**
   - Load testing completed
   - Scaling behavior verified
   - Failover mechanisms tested
   - Security scanning passed

3. **Integration Testing**
   - API endpoints tested
   - Database connections verified
   - External service integrations working
   - Monitoring and logging functional

Production Deployment Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pre-deployment**
   - Backup current model
   - Set maintenance mode if needed
   - Verify rollback procedures

2. **Deployment**
   - Deploy new version
   - Run health checks
   - Verify functionality
   - Monitor key metrics

3. **Post-deployment**
   - Remove maintenance mode
   - Monitor for errors
   - Validate performance
   - Update documentation

Best Practices
--------------

1. **Model Versioning**
   - Use semantic versioning
   - Maintain model registry
   - Track performance metrics per version

2. **Scaling Strategy**
   - Horizontal scaling for high throughput
   - Auto-scaling based on metrics
   - Load balancing across instances

3. **Security**
   - API authentication and authorization
   - Input validation and sanitization
   - Regular security updates

4. **Monitoring**
   - Real-time performance monitoring
   - Model drift detection
   - Comprehensive logging

5. **Disaster Recovery**
   - Automated backups
   - Multi-region deployment
   - Clear rollback procedures

This deployment guide provides a comprehensive foundation for taking your SITS-Former model from development to production, ensuring reliability, scalability, and maintainability.