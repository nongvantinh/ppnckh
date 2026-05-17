# Đồ án: Máy học dự đoán bệnh tiểu đường

Chủ đề: **Nghiên cứu và đánh giá hiệu quả của các thuật toán máy học trong việc dự đoán bệnh tiểu đường.**

Luận văn/báo cáo LaTeX (`main.tex`) theo cấu trúc IMRaD: **Tóm tắt + Keywords**, Giới thiệu (**RQ1/RQ2**, đóng góp dạng bullet), **Tổng quan** (so sánh định lượng + bảng trích dẫn), Dữ liệu và phương pháp (**pipeline TikZ**, reproducibility, chỉ số kèm FP/FN), Thực nghiệm (**bảng điểm + mean±std**), Thảo luận (**error analysis**, đạo đức/lâm sàng), Kết luận, **Phụ lục siêu tham số**, Tài liệu tham khảo (IEEE qua `biblatex`).

## Yêu cầu

- **TeX Live** (khuyến nghị đủ gói: `xelatex`, `biber`, `biblatex`)
- **LaTeX Workshop** (VS Code/Cursor), tùy chọn
- **Python + uv** (cho `resources/diabetes.ipynb`): [uv](https://docs.astral.sh/uv/) quản lý môi trường và gói

### Notebook thực nghiệm (`uv`)

```bash
cd /path/to/ppnckh
uv sync
uv run jupyter lab resources/diabetes.ipynb
# hoặc chạy headless:
uv run jupyter execute resources/diabetes.ipynb
```

- Dữ liệu Pima tải về `resources/data/` lần đầu (file `.csv` bị `.gitignore`; có thể chạy lại notebook để tải).
- SHAP (tùy chọn): `uv sync --extra shap`
- Tạo lại file `.ipynb` từ script (khi sửa generator): `python3 scripts/generate_diabetes_notebook.py`
- Sau **Run All**, notebook gọi `resources/export_run.py` → dữ liệu trong `resources/outputs/latest/` (CSV, `manifest.json`, thư mục `figures/`) và bản snapshot `resources/outputs/runs/<timestamp>/`.

## Biên dịch

Dự án dùng **XeLaTeX** + **biber** (xem `latexmk` / `.latexmkrc` nếu có).

```bash
cd /home/ubuntu/Projects/ppnckh
latexmk -xelatex main.tex
```

Hoặc:

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

## Cấu trúc thư mục

| Mục | Mô tả |
|-----|--------|
| `main.tex` | Nội dung báo cáo (bìa trường, mục lục, các mục IMRaD) |
| `references.bib` | Tài liệu tham khảo (IEEE) |
| `image/` | Logo bìa và hình minh họa (ROC, confusion matrix, …) |
| `resources/` | Notebook, ứng dụng demo, tài liệu nguồn (`.ipynb`, `.docx`, …) |

## Ghi chú

- Điền số liệu thực nghiệm vào bảng trong mục 4 và cập nhật Tóm tắt (~150–250 từ, không trích dẫn).
- Trích dẫn mẫu cho tổng quan/dữ liệu: `smith1988`, `uci-pima`, `pedregosa2011` trong `references.bib`.

Prompt cho Google genmini giúp sửa tiếng việt cho trôi chảy. Lưu ý, chỉ nên copy 2 đoạn ngắn vào LLM để nó sửa xúc tích nhất. nội dung càng dài LLM càng sơ lược bớt đầu ra.
```bash
Giúp mình sửa đoạn sau, mình muốn nó nghe tự nhiên, trôi chảy hơn khi đọc, câu văn phải chắc chắn và chuyên nghiệp, phù hợp trong bối cảnh nghiên cứu. Đảm bảo giống như là 1 người việt viết, không sử dụng dấu gạch ngang (—) để nối câu, hãy thay bằng các từ nối hoặc cấu trúc câu thuần Việt, bạn phải đưa cho mình đầu ra với cú pháp latex và để code LaTeX trong khối mã:
```