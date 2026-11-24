# CPE 342 Term Project — Technical Report Package (`project/report/`)

แพ็กเกจรายงานทางวิชาการฉบับภาษาไทยสำหรับโครงงาน **CPE 342 Machine Learning Term Project: ระบบอัจฉริยะวิเคราะห์พฤติกรรมผู้เล่น Karena (Karena Player Intelligence System)** โดยกลุ่ม **ML004** (ภาคเรียนที่ 1 ปีการศึกษา 2569 / Semester 1/2026) จัดทำด้วย XeLaTeX (ฟอนต์ `Sarabun` + `Consolas`) ปราศจากหัวกระดาษ (Clean Headerless Academic/Thesis Layout) และอ้างอิงโครงสร้างงานวิชาการระดับปริญญาตรีจากคลังอ้างอิง `past-reference/`

## โครงสร้างไฟล์

```text
project/report/
├── .latexmkrc                       # ค่าเริ่มต้นคอมไพล์ XeLaTeX ลง temp dir ไม่ทิ้งไฟล์ขยะใน repo
├── README.md                        # เอกสารคู่มือชุดนี้
├── evidence/
│   └── README.md                    # ดัชนีตรวจสอบย้อนกลับระหว่างตัวเลขในรายงานกับไฟล์หลักฐาน
├── frontpage/
│   ├── frontpage.tex                # หน้าปกรายงานภาษาไทยอย่างเป็นทางการ (โลโก้ KMUTT + รายชื่อสมาชิกกลุ่ม ML004)
│   ├── frontpage.example.tex        # ตัวอย่างต้นแบบจาก CPE333
│   └── assets/
│       ├── KMUTT_Logo.png
│       └── KMUTT_Logo.svg
├── assets/                          # ภาพกราฟวิเคราะห์ความละเอียดสูง 300 dpi (18 ภาพจาก Tasks 1–5)
├── past-reference/                  # เอกสารอ้างอิงรูปแบบรายงานวิชา CPE232
├── report.tex                       # ต้นฉบับรายงาน XeLaTeX ฉบับสมบูรณ์ (บทคัดย่อ, สารบัญ, บทที่ 1–9, บรรณานุกรม, ภาคผนวก ก)
└── output/
    └── pdf/
        └── CPE342_TermProject_Report.pdf
```

## คำสั่งคอมไพล์ PDF (ไม่สร้างไฟล์กลางลงใน Repository)

```sh
cd project/report
build_dir="$(mktemp -d -t cpe342-report)"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$build_dir" -jobname=CPE342_TermProject_Report report.tex
cp "$build_dir/CPE342_TermProject_Report.pdf" output/pdf/CPE342_TermProject_Report.pdf
rm -rf "$build_dir"
```

## ข้อสังเกตเรื่องการจัดส่งบน Git และ LEB2
- **รายงานฉบับ PDF (`CPE342_TermProject_Report.pdf`):** จัดส่งผ่านระบบ **LEB2** และเก็บต้นฉบับ LaTeX ไว้ในคลังส่วนตัว `Palapluem/cpe342-machine-learning-2026` (`project/report/`) แยกจากคลังซอร์สโค้ดรวมของกลุ่ม (`beambeambeam/ml00x`) เพื่อไม่ให้ปะปนกับท่อข้อมูลและสคริปต์ตรวจสอบการส่ง Kaggle ของทีม
- **ซอร์สโค้ด (10%):** จัดส่งผ่านลิงก์ GitHub Repository ตามโครงสร้าง `task1/` ถึง `task5/` แยกต่างหาก จึงไม่ได้คัดลอกโค้ดฉบับเต็มกว่า 100 หน้าลงในภาคผนวกของรายงาน แต่สรุปเป็นแผนผังสถาปัตยกรรม Checksum SHA-256 และแกนตรรกะเชิงอัลกอริทึมแทน
