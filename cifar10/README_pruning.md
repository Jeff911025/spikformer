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

### 2. 修改後的 `train.py`
整合了 pruning 機制到訓練流程中：

- 新增了 pruning 相關的命令行參數
- 在訓練循環中定期應用 pruning
- 自動更新模型和優化器

### 3. `example_pruning.py`
提供了一個完整的示例腳本來展示如何使用 pruning 機制。

## 使用方法

### 方法 1: 使用修改後的訓練腳本

```bash
# 啟用 pruning 的訓練
python train.py --enable-pruning --pruning-ratio 0.3 --pruning-epochs 50 --pruning-interval 10

# 其他參數
python train.py \
    --enable-pruning \
    --pruning-ratio 0.3 \
    --pruning-epochs 50 \
    --pruning-interval 10 \
    --epochs 200 \
    --batch-size 32 \
    --lr 0.01
```

### 方法 2: 使用示例腳本

```bash
# 基本使用
python example_pruning.py

# 自定義參數
python example_pruning.py \
    --pruning-ratio 0.3 \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.01
```

## Pruning 參數說明

- `--enable-pruning`: 啟用 pruning 機制
- `--pruning-ratio`: pruning 比例 (0-1)，預設 0.3
- `--pruning-epochs`: 開始 pruning 的 epoch，預設 50
- `--pruning-interval`: pruning 評估間隔，預設 10

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

可以擴展的分數計算方法：
- **變異數**: 計算 spike 頻率的變異數
- **熵**: 計算 spike 模式的熵
- **相關性**: 計算 channels 之間的相關性

## 注意事項

1. **模型兼容性**: 目前支援 SSA 和 MLP 模組的 pruning
2. **維度匹配**: Pruning 後需要確保相鄰層的維度匹配
3. **性能影響**: Pruning 可能會影響模型性能，需要根據具體任務調整 pruning 比例
4. **漸進式 Pruning**: 建議使用漸進式 pruning 而不是一次性大量 pruning

## 擴展建議

1. **更多分數指標**: 實現更多樣化的 channel 重要性評估方法
2. **結構化 Pruning**: 支援 head-wise 或 layer-wise 的結構化 pruning
3. **動態調整**: 根據訓練進度動態調整 pruning 策略
4. **恢復機制**: 實現被 pruning 的 channels 的恢復機制

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