# CPE 342 Term Project — Technical Report (`project/report/`)

แพ็กเกจรายงานทางวิชาการสำหรับโครงงาน **CPE342 Term Project: Karena Player Intelligence System** ของกลุ่ม **ML004** (ภาคเรียนที่ 1 ปีการศึกษา 2569) จัดทำด้วย XeLaTeX (ฟอนต์ `TH Sarabun New` `Scale=1.46` + `Consolas`) ปราศจากหัวกระดาษ (Clean Headerless Academic Layout) ตามรูปแบบเอกสารอ้างอิงรายวิชาของภาควิชาวิศวกรรมคอมพิวเตอร์ มจธ.

## โครงสร้างไฟล์

```text
project/report/
├── .latexmkrc                       # การตั้งค่า XeLaTeX ลง temp directory ไม่ทิ้งไฟล์ขยะใน repo
├── README.md                        # เอกสารแนะนำโครงสร้างรายงานและวิธีคอมไพล์
├── evidence/
│   └── README.md                    # ดัชนีอ้างอิงแหล่งข้อมูลและผลรันที่ใช้ประกอบรายงาน
├── frontpage/
│   ├── frontpage.tex                # หน้าปกรายงาน (โลโก้ KMUTT, CPE342 Term Project, สมาชิกกลุ่ม ML004)
│   ├── frontpage.example.tex        # ตัวอย่างต้นแบบเดิม
│   └── assets/
│       ├── KMUTT_Logo.png
│       └── KMUTT_Logo.svg
├── assets/                          # กราฟการวิเคราะห์ความละเอียด 300 dpi (18 ภาพจาก Tasks 1–5)
├── past-reference/                  # เอกสารอ้างอิงรูปแบบรายงานและสำนวนวิชาการภาษาไทย
├── report.tex                       # ต้นฉบับรายงาน XeLaTeX ฉบับสมบูรณ์ (สารบัญ, บทที่ 1–8, บรรณานุกรม)
└── output/
    └── pdf/
        └── CPE342_TermProject_Report.pdf
```

## คำสั่งคอมไพล์ PDF

```sh
cd project/report
build_dir="$(mktemp -d -t cpe342-report)"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$build_dir" -jobname=CPE342_TermProject_Report report.tex
cp "$build_dir/CPE342_TermProject_Report.pdf" output/pdf/CPE342_TermProject_Report.pdf
rm -rf "$build_dir"
```

## การแยกจัดเก็บซอร์สโค้ดและรายงาน
- **รายงานฉบับ PDF (`CPE342_TermProject_Report.pdf`):** จัดส่งผ่านระบบ LEB2 และเก็บต้นฉบับ LaTeX ไว้ใน `cpe342-machine-learning-2026/project/report/`
- **ซอร์สโค้ด (`10%`):** จัดส่งผ่าน GitHub Repository แยกตามโฟลเดอร์งาน (`task1/` ถึง `task5/`) แยกส่วนจากเอกสารรายงาน
