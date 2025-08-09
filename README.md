# Stock-market-forecasting - Dự báo xu hướng giá cổ phiếu

Đây là dự án **Dự báo xu hướng giá cổ phiếu** sử dụng các phương pháp học máy, đặc biệt là mạng LSTM, để phân tích và dự báo xu hướng chứng khoán trong ngày (intraday) dựa trên dữ liệu giá mở cửa và đóng cửa. Hệ thống hỗ trợ xử lý dữ liệu, gán nhãn, huấn luyện mô hình và đánh giá kết quả dự báo trên thị trường thực tế.

## 🎯 Mục tiêu

Mục tiêu chính của đề tài:
- **Thu thập dữ liệu**: Thu thập và xử lý dữ liệu lịch sử giá mở/đóng của cổ phiếu trong ngày.
- **Xử lý nhãn**: Tạo nhãn phân loại (labels) dựa trên biến động giá để phục vụ bài toán dự báo.
- **Xây dựng mô hình**: Xây dựng và huấn luyện mô hình LSTM dự đoán xu hướng tăng/giảm hoặc phân loại mức độ biến động ngắn hạn.
- **Đánh giá mô hình**: Đánh giá hiệu quả mô hình trên dữ liệu thực tế, hỗ trợ cải tiến các chiến lược giao dịch tự động.
- **Module hóa các chức năng**: Dễ dàng mở rộng và tùy chỉnh cho các loại dữ liệu hoặc thuật toán khác trong lĩnh vực tài chính.


## 🔬 Các mô hình được so sánh

Hệ thống triển khai 6 mô hình cơ bản:

1. **IntraDay-240,3-LSTM** - Model: LSTM, Feature: (ir, cr, or), Window: 240
2. **IntraDay-240,1-LSTM** - Model: LSTM, Feature: (ir), Window: 240
3. **NextDay-240,1-LSTM** - Model: LSTM, Feature: (cr), Window: 240
4. **IntraDay-240,3-RF** - Model: RF, Feature: (ir, cr, or), Window: 240
5. **IntraDay-240,1-RF** - Model: RF, Feature: (ir), Window: 240
6. **NextDay-240,1-RF** - Model: RF, Feature: (cr), Window: 240

## 📊 Bộ dữ liệu

Dự án sử dụng 3 bộ dữ liệu chuẩn quốc tế:

- **[GSM8K](https://github.com/openai/grade-school-math)** - 8,500 bài toán toán học chất lượng cao cho học sinh tiểu học
- **[TATQA](https://github.com/NExTplusplus/TAT-QA)** - 16,552 câu hỏi về báo cáo tài chính từ 2,757 ngữ cảnh thực tế
- **[TABMWP](https://github.com/lupantech/PromptPG)** - 38,431 bài toán yêu cầu lý luận trên cả văn bản và bảng biểu

## 🚀 Cài đặt và thiết lập

### Yêu cầu hệ thống
- Python 3.12.10+

### 1. Clone repository

```bash
git https://github.com/Kiwikaito04/Stock-market-forecasting.git
cd Stock-market-forecasting
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Thiết lập biến môi trường

```bash
cp env.example .env
```

Chỉnh sửa file `.env` với thông tin cần thiết:

```env
# Năm bắt đầu và năm kết thúc
START_YEAR=1990
END_YEAR=2018

# Cửa sổ trượt (3 năm train - 1 năm test)
WINDOW_SIZE=3

# Thư mục lưu trữ dữ liệu
DATA_FOLDER='dataset'

# Thiết lập SEED để đảm bảo bảo toàn kết quả
SEED=727
```


## 💻 Cách sử dụng

### Chạy thí nghiệm đơn lẻ

```bash
# Chạy một thực nghiệm LSTM 3 đặc trưng
python Intraday-240,3-LSTM.py
```

### Chạy thí nghiệm và xem biểu đồ

```bash
# Chạy tất cả
python __main__.py --all

# Chạy nhóm 'paper'   
python __main__.py --group paper

# Chạy cụ thể       
python __main__.py --scripts intra_3_lstm intra_3_rf
```

### Xem biểu đồ

```bash
# Tạo biểu đồ so sánh các phương pháp

# Tạo biểu đồ chi phí
```

## 📁 Cấu trúc dự án

```
MAS/
├── __main__.py              # Script chính để tạo visualization và phân tích
├── mathqa.py               # Module utilities cho mathematical operations  
├── few_shot_PoT.py         # Implementation few-shot Program of Thoughts
├── requirements.txt        # Dependencies cần thiết
├── env.example            # Template cấu hình môi trường
├── README_DATASET.md      # Documentation chi tiết về datasets
├── datasets/              # Raw datasets
│   ├── GSM8K/            # Grade School Math dataset
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   ├── test_socratic.jsonl
│   │   ├── train_socratic.jsonl
│   │   └── example_model_solutions.jsonl
│   ├── TATQA/            # Table and Text QA dataset  
│   │   ├── tatqa_dataset_train.json
│   │   ├── tatqa_dataset_dev.json
│   │   ├── tatqa_dataset_test.json
│   │   └── tatqa_dataset_test_gold.json
│   └── TABMWP/           # Tabular Math Word Problems
│       ├── algorithm.png
│       ├── dataset.png
│       ├── prediction.png
│       ├── promptpg.png
│       └── tabmwp/
├── mint/                 # Core framework package
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── Zero_shot.py      # Zero-shot implementation
│   ├── CoT.py           # Chain of Thought implementation  
│   ├── PoT.py           # Program of Thoughts implementation
│   ├── PaL.py           # Program-aided Language implementation
│   ├── chart.py         # Chart generation utilities
│   ├── dataset_to_langsmith.py  # LangSmith integration tools
│   └── testing/         # Testing modules
│       └── CoT.py
├── EDA/                 # Exploratory Data Analysis notebooks
│   ├── GSM8K_EDA.ipynb
│   ├── TATQA_EDA.ipynb
│   └── TABMWP_EDA.ipynb
├── results/             # Processed results từ experiments
│   ├── CoT_GSM8K.json
│   ├── CoT_TABMWP.json
│   ├── CoT_TATQA.json
│   ├── MultiAgent_GSM8K.json
│   ├── MultiAgent_TABMWP.json
│   ├── MultiAgent_TATQA.json
│   ├── PaL_GSM8K.json
│   ├── PaL_TATQA.json
│   ├── PoT_GSM8K.json
│   ├── PoT_TABMWP.json
│   ├── PoT_TATQA.json
│   ├── Zero-shot_GSM8K.json
│   ├── Zero-shot_TABMWP.json
│   └── Zero-shot_TATQA.json
└── save_log/           # Chi tiết logs với timestamps
    ├── CoT_results_gsm8k_26-07-2025_16:36:03_500samples.json
    ├── CoT_results_tabmwp_26-07-2025_18:12:37_500samples.json
    ├── CoT_results_tatqa_26-07-2025_17:01:36_500samples.json
    ├── MultiAgent_results_gsm8k_27-07-2025_01:40:00_500samples.json
    ├── MultiAgent_results_tabmwp_27-07-2025_00:54:48_500samples.json
    ├── MultiAgent_results_tatqa_27-07-2025_02:32:28_500samples.json
    ├── PaL_results_gsm8k_26-07-2025_19:01:30_500samples.json
    ├── PaL_results_tabmwp_26-07-2025_18:33:35_500samples.json
    ├── PaL_results_tatqa_26-07-2025_19:18:41_500samples.json
    ├── PoT_results_gsm8k_26-07-2025_14:34:04_500samples.json
    ├── PoT_results_tabmwp_26-07-2025_15:37:25_500samples.json
    ├── PoT_results_tatqa_26-07-2025_14:52:34_500samples.json
    ├── Zero-shot_results_gsm8k_26-07-2025_13:26:24_500samples.json
    ├── Zero-shot_results_tabmwp_26-07-2025_15:45:06_500samples.json
    └── Zero-shot_results_tatqa_26-07-2025_13:36:44_500samples.json
```


## 🔧 Tính năng nâng cao

### Sandbox Code Execution
Hệ thống sử dụng môi trường sandbox an toàn để thực thi code Python:
- **Isolated Environment**: Code được chạy trong môi trường cô lập, đảm bảo an toàn hệ thống
- **Timeout Protection**: Giới hạn thời gian thực thi để tránh infinite loops
- **Resource Limiting**: Kiểm soát việc sử dụng memory và CPU
- **Safe Libraries**: Chỉ cho phép import các thư viện math/science an toàn
- **Error Recovery**: Tự động phát hiện và sửa lỗi syntax với retry mechanism

### Advanced Visualization
Framework cung cấp hệ thống visualization mạnh mẽ:
- **Dual-axis plots**: Hiển thị accuracy và metrics khác cùng lúc
- **Multi-dataset comparison**: So sánh hiệu suất trên nhiều datasets
- **Custom color schemes**: Phân biệt rõ ràng giữa các methods với highlight colors
- **Statistical annotations**: Hiển thị giá trị chính xác trên biểu đồ
- **Publication-ready charts**: Export high-quality visualizations

### LangSmith Integration
Tích hợp toàn diện với LangSmith ecosystem:
- **Experiment Tracking**: Tự động ghi log tất cả các lần chạy với metadata
- **Cost Monitoring**: Theo dõi chi phí API calls theo thời gian thực
- **Performance Analytics**: Phân tích độ chính xác và latency của từng phương pháp
- **Dataset Versioning**: Quản lý versions của datasets và results
- **Traceability**: Full tracing từ input đến output cho debugging

### Error Handling & Debugging
Hệ thống debug thông minh với nhiều layers:
- **Syntax Error Detection**: Tự động phát hiện và sửa lỗi Python syntax
- **Logic Error Recovery**: Attempt to fix logical errors in generated code
- **Retry Mechanism**: Giới hạn số lần debug để tránh vòng lặp vô hạn
- **Detailed Logging**: Log chi tiết cho mỗi bước để phân tích vấn đề
- **Graceful Degradation**: Fallback strategies khi code execution fails


## 📊 Đánh giá hiệu suất

Hệ thống được đánh giá trên các tiêu chí:
- **Accuracy**: Tỷ lệ trả lời đúng
- **Cost**: Chi phí API calls
- **Latency**: Thời gian phản hồi
- **Token Usage**: Số token sử dụng


## 🔗 Liên hệ

- GitHub: [@Kiwikaito04](https://github.com/Kiwikaito04/), [@SunShine-chi](https://github.com/SunShine-chi)
- Project Link: [https://github.com/Kiwikaito04/Stock-market-forecasting](https://github.com/Kiwikaito04/Stock-market-forecasting)


## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) cho GPT models
- [LangChain](https://langchain.com/) cho framework
- [LangSmith](https://smith.langchain.com/) cho tracing và evaluation
- Các tác giả của GSM8K, TATQA, và TABMWP datasets