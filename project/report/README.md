# CPE 342 Term Project — Technical Report (`project/report/`)

แพ็กเกจรายงานทางวิชาการฉบับสมบูรณ์สำหรับโครงงาน **CPE342 Term Project: Karena Player Intelligence System** ของกลุ่ม **ML004** (ภาคเรียนที่ 1 ปีการศึกษา 2569) จัดทำด้วย XeLaTeX (ฟอนต์ `TH Sarabun New` `Scale=1.46` + `Consolas`) ปราศจากหัวกระดาษ ตามรูปแบบเอกสารอ้างอิงรายวิชาภาควิชาวิศวกรรมคอมพิวเตอร์ มจธ.

## โครงสร้างไฟล์

```text
project/report/
├── .latexmkrc                       # ตั้งค่า XeLaTeX ให้ส่ง aux ไปยัง TMPDIR และ PDF ไปยัง output/pdf/
├── README.md                        # เอกสารแนะนำโครงสร้างและวิธีคอมไพล์รายงาน
├── CPE342_TermProject_Report.tex    # ต้นฉบับรายงาน XeLaTeX ฉบับสมบูรณ์ (สารบัญ, บทที่ 1–8, บรรณานุกรม)
├── frontpage/
│   ├── frontpage.tex                # หน้าปกรายงาน (โลโก้ KMUTT, CPE342 Term Project, สมาชิกกลุ่ม ML004)
│   └── assets/
│       └── KMUTT_Logo.png
├── assets/                          # กราฟวินิจฉัยความละเอียด 300 dpi (19 ภาพจาก Tasks 1–5 และ Leaderboard)
└── output/
    └── pdf/
        └── CPE342_TermProject_Report.pdf
```

## คำสั่งคอมไพล์ PDF

```sh
cd project/report
latexmk CPE342_TermProject_Report.tex
```

ไฟล์ `.latexmkrc` จะรัน `xelatex` และแยกไฟล์ชั่วคราว (`.aux`, `.log`, `.out`, `.toc`, `.xdv`) ไปยัง `$TMPDIR/cpe342-report-aux` โดยอัตโนมัติ พร้อมบันทึกไฟล์ผลลัพธ์ลงที่ `output/pdf/CPE342_TermProject_Report.pdf`

## การแยกจัดเก็บซอร์สโค้ดและรายงาน
- **รายงานฉบับ PDF (`CPE342_TermProject_Report.pdf`):** จัดส่งผ่านระบบ LEB2 และเก็บต้นฉบับ LaTeX ไว้ใน `cpe342-machine-learning-2026/project/report/` (และซิงก์ใน `beambeambeam/ml00x/report/`)
- **ซอร์สโค้ด (`10%`):** จัดส่งผ่าน GitHub Repository แยกตามโฟลเดอร์งาน (`task1/` ถึง `task5/`)
