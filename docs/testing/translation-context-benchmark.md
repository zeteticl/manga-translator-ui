# 翻譯上下文 TEST

此工具只執行 OCR、圖片分析及文字翻譯，結果寫入獨立的 `result/`。

## 三組

1. 普通：現有普通提示及前 3 張原文／譯文。
2. Extreme：現有 Extreme 提示，前 3 張及後 2 張資料。
3. Description + dialogue：固定提示，獨立調整 N、Y。N 是前 N 張和後 N 張的圖片 description；Y 是前 Y 張和後 Y 張的原文 OCR 台詞。當頁 description 和台詞固定提供，沒有歷史譯文。Description 使用固定的圖片單獨描述提示，不提供 OCR，要求模型不引述或概括對白及最多 90 英文詞；儲存時最多 600 字元。詞數及圖內文字禁讀是提示要求，未作硬性驗證。OCR 台詞可能包含標題；主要評分排除標題。

N、Y 各測 0、1、3、5，共 16 組。N=Y=0 是只有當頁資料的對照。

所有翻譯組別共用本地模型、取樣參數、完整回應 schema、ID 驗證及按 ID 排序的傳輸正規化。翻譯每次請求最多 2048 output tokens、180 秒；圖片單獨描述最多 768 tokens、180 秒，固定供第三組所有配置共用。Extreme 使用現有結構化分析快取。產品程式及使用者設定不修改。這是受控翻譯品質測試；原版未加 schema 的完成率在之前的報告另列。

## 頁面及流程

- 原圖：`ドラゴンクエスト列伝 ロトの紋章 Returns`。
- 參考資料：page_6–19、page_41–54，足夠提供每邊 5 張。
- 普通／Extreme 從 page_8、43 開始逐張翻譯以建立真實歷史，主要評分 page_11–14、46–49。
- 選組合：page_12、13、47、48，41 組主要文字。
- 保留驗證：page_11、14、46、49，35 組主要文字，不用於初步選組合。
- 選取第三組前兩名後，驗證及以第二個 seed 重跑驗證頁面。
- 品質由 Codex 根據原文及原圖作匿名評分。0=不能使用，1=意思／關係／漏譯有實質錯誤，2=意思保留但語氣／流暢度／措辭有小問題，3=正確自然。同一句相同候選只評一次，分數供所有產生相同文字的組合共用。
- 依選組合頁面的總分排序，再比較實質錯誤數和 input tokens；不能以文字不同率代替準確度。

每次請求保存原回應、參數、耗時與 tokens；圖片以 hash 代替，沒有保存憑證。快取可重用，但檢查來源及程式 hash；續跑不覆寫之前請求紀錄。

## CLI

```powershell
$source = 'E:\github\fuckBookWalker\babies\ドラゴンクエスト列伝 ロトの紋章 Returns'
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage prepare --reuse-cache result\extreme_text_eval_20261008
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage descriptions
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage baselines
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage grid --round discovery
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage review --round discovery
# 只讀 review/discovery/blind.json 作評分，先寫 judgments.json 再揭示 key。
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage rank --round discovery
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage grid --round validation
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage review --round validation
# 記錄驗證評分後才 rank。
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage rank --round validation
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_translation_context.py --source $source --stage grid --round confirmation
```

第二輪需另外建立 review 和 judgments，再執行 rank。`judgments.json` 是陣列，每個候選包含 `id`、0–3 的 `score`、非空的 `reason`。

單元測試：`python -m pytest test/test_translation_context_benchmark.py`。

## 解讀限制

普通與 Extreme 保留各自提示，第三組使用固定提示，因此三組整體差異不能只歸因於視覺資訊。只有第三組的 N、Y 掃描保持提示固定。description 本身可能不準確，也可能違反不引用圖中文字的提示；本測試固定第三組描述內容，評估現有描述流程的數量而非更換描述提示，不能當作完全隔離語言訊息的視覺實驗。最佳只代表這個模型、這批樣本及測試數值中的最佳；單一 seed 或小樣本不足以證明普遍最佳。圖片分析成本和翻譯成本須分開列出，快取耗時不能冒充首次處理時間。

2026-10-09 的已完成結果在 `result/context_grid_20261009_v4/report.md`，逐句評分在同資料夾的 `scored_translations.csv`。測試刻意保留模型的描述誤判及提示違規，另列限制；不在看到分數後修改描述重算排名。
