# Đồ án: Máy học dự đoán bệnh tiểu đường

Chủ đề: **Nghiên cứu và đánh giá hiệu quả của các thuật toán máy học trong việc dự đoán bệnh tiểu đường.**

Luận văn/báo cáo LaTeX (`main.tex`) theo cấu trúc IMRaD: **Tóm tắt + Keywords**, Giới thiệu (**RQ1/RQ2**, đóng góp dạng bullet), **Tổng quan** (so sánh định lượng + bảng trích dẫn), Dữ liệu và phương pháp (**pipeline TikZ**, reproducibility, chỉ số kèm FP/FN), Thực nghiệm (**bảng điểm + mean±std**), Thảo luận (**error analysis**, đạo đức/lâm sàng), Kết luận, **Phụ lục siêu tham số**, Tài liệu tham khảo (IEEE qua `biblatex`).

## Yêu cầu

- **TeX Live** (khuyến nghị đủ gói: `xelatex`, `biber`, `biblatex`)
- **LaTeX Workshop** (VS Code/Cursor), tùy chọn

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
