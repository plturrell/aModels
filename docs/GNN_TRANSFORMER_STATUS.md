# GNN and Transformer Features Status

## Current Status

### ✅ **Classes Are Importable**
- GNN and Transformer classes can be imported successfully
- The service reports them as "available" in health checks
- This is because the import errors are handled gracefully

### ❌ **Features Are NOT Functionally Working**
- **PyTorch is NOT installed** - Required for both GNN and Transformer
- **torch_geometric is NOT installed** - Required for GNN
- **transformers library is NOT installed** - Required for Transformer

## What This Means

### When You Try to Use GNN/Transformer:
1. **Class Import**: ✅ Works (classes can be imported)
2. **Instantiation**: ⚠️ Will work but with limited functionality
3. **Actual Learning**: ❌ Will fail when trying to use torch operations

### Current Behavior:
- The classes can be instantiated
- `HAS_PYG = False` and `HAS_TORCH = False` in the modules
- Methods that require torch will return `None` or raise errors
- Pattern learning will fall back to non-deep-learning methods

## To Enable Full Functionality

### Option 1: Uncomment in requirements-service.txt
```txt
# Uncomment these lines:
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0
```

### Option 2: Add to Dockerfile
```dockerfile
RUN pip install --no-cache-dir torch torch-geometric transformers
```

### Option 3: Install at runtime (not recommended)
```bash
docker exec training-service pip install torch torch-geometric transformers
```

## Verification

Current status can be verified by:
1. Health endpoint: `GET /health` - Shows `gnn_available: true` (but only means class importable)
2. Availability endpoints: `GET /patterns/gnn/available` - Returns `available: true`
3. **Actual functionality**: Will fail when trying to learn patterns with `use_gnn: true`

## Recommendation

**For Production**: Uncomment the optional dependencies in `requirements-service.txt` if you need GNN/Transformer features. Otherwise, the service works fine with basic pattern learning (non-deep-learning methods).

**For Testing**: The current setup is fine - tests can check availability, but actual deep learning features need the dependencies installed.

---

**Status**: ✅ Classes available, ⚠️ Functionality limited, ❌ Full features require dependencies

