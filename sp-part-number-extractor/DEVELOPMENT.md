# Part Number Extractor - Development Notes

## Project Structure Explained

### Core Modules

#### 1. Data Preparation (`src/data_preparation/`)
- **preprocessor.py**: Converts BOM rows to NER format
  - Tokenization using transformer tokenizers
  - Label encoding (O, B-PART, I-PART)
  - Padding and truncation
  
- **data_loader.py**: Dataset classes and I/O
  - BOMDataset for PyTorch
  - JSON loading/saving
  - CSV/Excel reading
  - Train/val/test splitting
  
- **augmentation.py**: Data augmentation techniques
  - Column shuffling
  - Noise addition
  - Part Number format variation

#### 2. Model (`src/model/`)
- **ner_model.py**: Transformer-based NER model
  - BERT/RoBERTa/DeBERTa backbone
  - Token classification head
  - Compatible with Hugging Face Trainer

#### 3. Training (`src/training/`)
- **trainer.py**: Training utilities
  - TrainingArguments creation
  - Model training with Hugging Face Trainer
  - Checkpointing and early stopping

#### 4. Evaluation (`src/evaluation/`)
- **metrics.py**: Evaluation metrics
  - Token-level F1 using seqeval
  - Part Number extraction accuracy
  - Error analysis tools

#### 5. Inference (`src/inference/`)
- **predictor.py**: Prediction engine
  - Single row prediction
  - Batch prediction
  - Confidence thresholding

### Scripts

- **train.py**: CLI training script
- **predict.py**: CLI prediction script
- **evaluate.py**: CLI evaluation script
- **interactive_label.py**: Interactive labeling tool
- **train_interactive.py**: Interactive training wizard
- **predict_interactive.py**: Interactive prediction tool
- **split_data.py**: Data splitting utility

### Configuration

YAML configs for different models:
- **bert_base.yaml**: BERT-base settings
- **roberta_base.yaml**: RoBERTa-base settings
- **deberta_v3.yaml**: DeBERTa-v3-base settings

## Development Workflow

### Phase 1: Data Preparation (Week 1-2)
1. Collect BOM files → `data/raw/`
2. Label using interactive tool → `data/labeled.json`
3. Split into train/val/test → `data/*.json`
4. (Optional) Apply augmentation

### Phase 2: Model Training (Week 3)
1. Start with BERT-base for quick iteration
2. Monitor training via logs
3. Evaluate on validation set
4. Adjust hyperparameters

### Phase 3: Optimization (Week 4)
1. Try RoBERTa or DeBERTa
2. Tune hyperparameters
3. Apply data augmentation
4. Achieve 95%+ accuracy

### Phase 4: Deployment (Week 5)
1. Test on real-world BOM files
2. Adjust confidence thresholds
3. Document edge cases
4. Create user guide

## Key Design Decisions

### 1. NER Approach (Token Classification)
**Why?**
- Handles variable column positions naturally
- Leverages pre-trained transformer knowledge
- Flexible: can extract multiple fields in future

**Alternatives considered:**
- Cell-level classification: Less context-aware
- Seq2Seq: Overkill for this task

### 2. Label Scheme (BIO Tagging)
**Labels:**
- O: Not part of Part Number
- B-PART: Beginning of Part Number
- I-PART: Inside Part Number

**Why BIO?**
- Standard in NER
- Handles multi-token Part Numbers
- Compatible with seqeval

### 3. Local Development Focus
**Why?**
- No cloud costs
- Data privacy
- Full control
- Easier debugging

## Performance Optimization Tips

### 1. Training Speed
- Use mixed precision (fp16)
- Gradient accumulation for effective larger batch size
- Smaller models for prototyping (BERT > RoBERTa > DeBERTa)

### 2. Inference Speed
- Batch prediction
- Model quantization (INT8)
- ONNX Runtime
- GPU utilization

### 3. Memory Management
- Reduce batch_size if OOM
- Use gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

## Common Issues and Solutions

### Issue 1: Low Accuracy on Part Numbers
**Possible causes:**
- Insufficient training data
- Part Number patterns not seen during training
- Imbalanced data

**Solutions:**
- Collect more diverse samples
- Apply data augmentation
- Adjust class weights

### Issue 2: False Positives
**Possible causes:**
- Other columns look similar to Part Numbers
- Model overconfident

**Solutions:**
- Add post-processing rules (regex filters)
- Adjust confidence threshold
- Ensemble with rule-based system

### Issue 3: Slow Inference
**Possible causes:**
- Large model
- CPU inference
- No batching

**Solutions:**
- Use smaller model
- Enable GPU
- Batch predictions
- Model distillation

## Future Enhancements

### Short-term (1-2 months)
- [ ] Multi-field extraction (Manufacturer, Package, etc.)
- [ ] Confidence calibration
- [ ] Rule-based post-processing
- [ ] Web UI for labeling

### Medium-term (3-6 months)
- [ ] Few-shot learning for new Part Number formats
- [ ] Active learning for data labeling
- [ ] Model compression (distillation)
- [ ] REST API server

### Long-term (6+ months)
- [ ] Multi-language support (Korean, Japanese)
- [ ] OCR integration for scanned BOMs
- [ ] Continuous learning pipeline
- [ ] Cloud deployment option

## Resources

### Papers
- BERT: https://arxiv.org/abs/1810.04805
- RoBERTa: https://arxiv.org/abs/1907.11692
- DeBERTa: https://arxiv.org/abs/2006.03654

### Documentation
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/docs
- seqeval: https://github.com/chakki-works/seqeval

### Tools
- TensorBoard: Monitor training
- Label Studio: Advanced labeling (optional)
- Weights & Biases: Experiment tracking (optional)

## Contact

For questions or contributions, please refer to README.md.
