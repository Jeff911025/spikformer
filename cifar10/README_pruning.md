# Spikformer Pruning 機制

這個目錄包含了為 Spikformer 模型實現的基於 spike map 的 pruning 機制。

## 概述

Pruning 機制通過分析神經元的 spike map 輸出，計算每個 channel 的活躍度分數，然後移除活躍度較低的 channels 來減少模型參數和計算量。

## 核心組件

### 1. `pruning_utils.py`
包含 pruning 的核心實現：

- **SpikeMapPruner**: 主要的 pruning 工具類
  - `register_hooks()`: 註冊 hooks 來捕獲 spike maps
  - `compute_channel_scores()`: 計算每個 channel 的分數
  - `select_channels_to_prune()`: 根據分數選擇要 pruning 的 channels
  - `apply_pruning()`: 應用 pruning 到模型

- **create_pruning_scheduler()**: 創建 pruning 比例調度器

### 2. `train_prune_finetune.py`（推薦完整流程腳本）

#### 支援的功能：
- **全流程**（預設）：訓練 → pruning → finetune
- **只做 pruning**：直接載入訓練好模型，進行 pruning
- **只做 finetune**：直接載入 pruning 後模型，finetune
- **只做 evaluate**：可分別評估 unpruned/pruned/finetuned 模型

#### Workflow 控制參數：
- `--do-train`：是否執行訓練（預設True）
- `--do-prune`：是否執行 pruning（預設True）
- `--do-finetune`：是否執行 finetune（預設True）
- `--load-unpruned`：直接載入訓練好模型（跳過訓練）
- `--load-pruned`：直接載入 pruning 後模型（跳過訓練和 pruning）
- `--eval-unpruned`：只評估訓練好（未 prune）模型
- `--eval-pruned`：只評估 pruning 後模型
- `--eval-finetuned`：只評估 finetuned 模型

#### 使用範例：

**全流程（預設）**
```bash
python train_prune_finetune.py
```

**只做 pruning**
```bash
python train_prune_finetune.py --do-train False --do-prune True --do-finetune False --load-unpruned ./output/train_prune_finetune/xxx/model_best.pth
```

**只做 finetune**
```bash
python train_prune_finetune.py --do-train False --do-prune False --do-finetune True --load-pruned ./output/train_prune_finetune/xxx/model_pruned.pth
```

**只做評估**
```bash
python train_prune_finetune.py --eval-unpruned --load-unpruned ./output/train_prune_finetune/xxx/model_best.pth
python train_prune_finetune.py --eval-pruned --load-pruned ./output/train_prune_finetune/xxx/model_pruned.pth
python train_prune_finetune.py --eval-finetuned
```

### 3. `example_pruning.py`
提供了一個簡單的訓練+pruning示例腳本，適合快速測試。

### 4. 修改後的 `train.py`
整合了 pruning 機制到訓練流程中：
- 新增了 pruning 相關的命令行參數
- 在訓練循環中定期應用 pruning
- 自動更新模型和優化器

## Pruning 流程

1. **Spike Map 捕獲**: 通過 hooks 捕獲每個模組的 spike map 輸出
2. **分數計算**: 壓縮時間維度，計算每個 channel 的平均活躍度
3. **Channel 選擇**: 根據分數選擇活躍度最低的 channels 進行 pruning
4. **模型更新**: 移除選定的 channels，更新模型結構
5. **優化器重構**: 為新的模型結構重新創建優化器

## 分數計算方法

目前實現的分數計算基於：
- **平均 Spike 頻率**: 計算每個 channel 在時間和空間維度上的平均 spike 頻率
- **低分數優先**: 選擇 spike 頻率最低的 channels 進行 pruning

## 注意事項

1. **模型兼容性**: 目前支援 SSA 和 MLP 模組的 pruning
2. **維度匹配**: Pruning 後需要確保相鄰層的維度匹配
3. **性能影響**: Pruning 可能會影響模型性能，需要根據具體任務調整 pruning 比例
4. **漸進式 Pruning**: 建議使用漸進式 pruning 而不是一次性大量 pruning

## 故障排除

### 常見問題

1. **Import 錯誤**: 確保所有依賴包已正確安裝
2. **維度不匹配**: 檢查模型結構，確保 pruning 後的維度匹配
3. **性能下降**: 降低 pruning 比例或調整 pruning 時機
4. **記憶體不足**: 減少 batch size 或使用梯度累積

### 調試建議

1. 使用較小的 pruning 比例開始測試
2. 監控 pruning 前後的模型參數數量
3. 記錄 pruning 對準確率的影響
4. 使用可視化工具分析 spike map 模式 