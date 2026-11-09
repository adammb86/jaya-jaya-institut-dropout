"""Prototype prediksi risiko dropout mahasiswa Jaya Jaya Institut.

Jalankan dengan: streamlit run app.py
"""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


def find_file(name):
    """Cari file model di folder model/ atau di root proyek."""
    for path in (BASE_DIR / "model" / name, BASE_DIR / name):
        if path.exists():
            return path
    return BASE_DIR / "model" / name


MODEL_PATH = find_file("rf_dropout_model.joblib")
META_PATH = find_file("model_metadata.json")

COURSE_MAP = {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
              8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
              9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
              9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
              9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
              9773: "Journalism and Communication", 9853: "Basic Education",
              9991: "Management (evening)"}
APP_MODE_MAP = {1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
                5: "1st phase - special contingent (Azores)", 7: "Holders of other higher courses",
                10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
                16: "1st phase - special contingent (Madeira)", 17: "2nd phase - general contingent",
                18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99 b2",
                27: "Ordinance No. 533-A/99 b3", 39: "Over 23 years old", 42: "Transfer",
                43: "Change of course", 44: "Technological specialization diploma",
                51: "Change of institution/course", 53: "Short cycle diploma holders",
                57: "Change of institution/course (International)"}
YES_NO = {1: "Ya", 0: "Tidak"}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def risk_level(prob, thresholds):
    if prob >= thresholds["Tinggi"]:
        return "Tinggi"
    if prob >= thresholds["Sedang"]:
        return "Sedang"
    return "Rendah"


def warning_signs(row):
    """Sinyal peringatan sederhana berdasarkan temuan EDA, untuk membantu konselor."""
    signs = []
    if row["Tuition_fees_up_to_date"] == 0:
        signs.append("UKT belum lunas tepat waktu (tingkat dropout kelompok ini sekitar 87%).")
    if row["Debtor"] == 1:
        signs.append("Memiliki tunggakan/utang ke institusi.")
    for sem, label in [("1st", "1"), ("2nd", "2")]:
        enrolled = row[f"Curricular_units_{sem}_sem_enrolled"]
        approved = row[f"Curricular_units_{sem}_sem_approved"]
        if enrolled > 0 and approved / enrolled < 0.5:
            signs.append(f"Hanya {approved} dari {enrolled} unit kurikuler lulus di semester {label}.")
        if enrolled > 0 and row[f"Curricular_units_{sem}_sem_grade"] < 10:
            signs.append(f"Rata-rata nilai semester {label} di bawah 10.")
    if row["Age_at_enrollment"] >= 25:
        signs.append("Masuk kuliah di usia 25 tahun ke atas (kelompok dengan dropout di atas 50%).")
    if row["Scholarship_holder"] == 0 and row["Tuition_fees_up_to_date"] == 0:
        signs.append("Bukan penerima beasiswa padahal ada kendala pembayaran. Pertimbangkan bantuan finansial.")
    return signs


st.set_page_config(page_title="Prediksi Dropout - Jaya Jaya Institut", page_icon="🎓", layout="wide")
model, meta = load_model()
FEATURES = meta["features"]
TH = meta["risk_thresholds"]

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.caption("Jaya Jaya Institut. Model Random Forest dilatih dari data mahasiswa Graduate dan Dropout "
           f"(akurasi {meta['test_metrics']['accuracy']:.1%}, F1 {meta['test_metrics']['f1']:.3f}, "
           f"ROC AUC {meta['test_metrics']['roc_auc']:.3f} pada data uji).")

tab_single, tab_batch, tab_about = st.tabs(["Prediksi satu mahasiswa", "Prediksi banyak mahasiswa (CSV)", "Tentang model"])

with tab_single:
    with st.form("form_mahasiswa"):
        st.subheader("Data pendaftaran dan demografi")
        c1, c2, c3 = st.columns(3)
        course = c1.selectbox("Program studi", list(COURSE_MAP), format_func=COURSE_MAP.get,
                              index=list(COURSE_MAP).index(9147))
        app_mode = c2.selectbox("Jalur pendaftaran", list(APP_MODE_MAP), format_func=APP_MODE_MAP.get)
        gender = c3.radio("Gender", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan", horizontal=True)
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Usia saat masuk", 17, 70, 20)
        admission = c2.number_input("Nilai masuk (0-200)", 0.0, 200.0, 127.0, step=0.5)
        prev_grade = c3.number_input("Nilai kualifikasi sebelumnya (0-200)", 0.0, 200.0, 133.0, step=0.5)

        st.subheader("Kondisi finansial")
        c1, c2, c3 = st.columns(3)
        tuition = c1.radio("UKT lunas tepat waktu?", [1, 0], format_func=YES_NO.get, horizontal=True)
        debtor = c2.radio("Memiliki utang?", [0, 1], format_func=YES_NO.get, horizontal=True)
        scholar = c3.radio("Penerima beasiswa?", [0, 1], format_func=YES_NO.get, horizontal=True)

        st.subheader("Performa akademik")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Semester 1**")
            e1 = st.number_input("Unit diambil (enrolled)", 0, 30, 6, key="e1")
            ev1 = st.number_input("Jumlah evaluasi", 0, 50, 8, key="ev1")
            a1 = st.number_input("Unit lulus (approved)", 0, 30, 5, key="a1")
            g1 = st.number_input("Rata-rata nilai (0-20)", 0.0, 20.0, 12.0, step=0.1, key="g1")
        with s2:
            st.markdown("**Semester 2**")
            e2 = st.number_input("Unit diambil (enrolled)", 0, 30, 6, key="e2")
            ev2 = st.number_input("Jumlah evaluasi", 0, 50, 8, key="ev2")
            a2 = st.number_input("Unit lulus (approved)", 0, 30, 5, key="a2")
            g2 = st.number_input("Rata-rata nilai (0-20)", 0.0, 20.0, 12.0, step=0.1, key="g2")
        submitted = st.form_submit_button("Prediksi risiko", type="primary", use_container_width=True)

    if submitted:
        errors = [f"Unit lulus semester {s} tidak boleh melebihi unit yang diambil."
                  for s, a, e in [(1, a1, e1), (2, a2, e2)] if a > e]
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            row = {"Curricular_units_1st_sem_enrolled": e1, "Curricular_units_1st_sem_evaluations": ev1,
                   "Curricular_units_1st_sem_approved": a1, "Curricular_units_1st_sem_grade": g1,
                   "Curricular_units_2nd_sem_enrolled": e2, "Curricular_units_2nd_sem_evaluations": ev2,
                   "Curricular_units_2nd_sem_approved": a2, "Curricular_units_2nd_sem_grade": g2,
                   "Tuition_fees_up_to_date": tuition, "Debtor": debtor, "Scholarship_holder": scholar,
                   "Age_at_enrollment": age, "Gender": gender, "Admission_grade": admission,
                   "Previous_qualification_grade": prev_grade, "Course": course, "Application_mode": app_mode}
            X = pd.DataFrame([row])[FEATURES]
            prob = float(model.predict_proba(X)[0, 1])
            level = risk_level(prob, TH)
            color = {"Tinggi": "red", "Sedang": "orange", "Rendah": "green"}[level]

            m1, m2 = st.columns(2)
            m1.metric("Probabilitas dropout", f"{prob:.1%}")
            m2.markdown(f"### Tingkat risiko: :{color}[{level}]")
            st.progress(prob)
            if level == "Tinggi":
                st.error("Mahasiswa ini perlu diprioritaskan untuk bimbingan khusus (dosen wali dan konseling).")
            elif level == "Sedang":
                st.warning("Pantau perkembangan mahasiswa ini dan jadwalkan sesi konsultasi akademik.")
            else:
                st.success("Risiko dropout rendah. Lanjutkan pemantauan rutin.")
            signs = warning_signs(row)
            if signs:
                st.markdown("**Sinyal peringatan yang terdeteksi:**")
                for s in signs:
                    st.markdown(f"- {s}")

with tab_batch:
    st.write("Unggah file CSV dengan format yang sama seperti dataset asli (pemisah `;` atau `,`). "
             "Minimal harus memuat 17 kolom fitur berikut:")
    st.code(", ".join(FEATURES), language=None)
    template = pd.DataFrame(columns=FEATURES)
    st.download_button("Unduh template CSV", template.to_csv(index=False, sep=";"), "template_mahasiswa.csv", "text/csv")
    uploaded = st.file_uploader("File CSV mahasiswa", type="csv")
    if uploaded is not None:
        raw = uploaded.getvalue().decode("utf-8-sig")
        sep = ";" if raw.splitlines()[0].count(";") > raw.splitlines()[0].count(",") else ","
        data = pd.read_csv(io.StringIO(raw), sep=sep)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing)}")
        else:
            result = data.copy()
            result["dropout_probability"] = model.predict_proba(data[FEATURES])[:, 1]
            result["risk_level"] = result["dropout_probability"].apply(lambda p: risk_level(p, TH))
            result["Course_name"] = result["Course"].map(COURSE_MAP)
            counts = result["risk_level"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total mahasiswa", len(result))
            c2.metric("Risiko tinggi", int(counts.get("Tinggi", 0)))
            c3.metric("Risiko sedang", int(counts.get("Sedang", 0)))
            c4.metric("Risiko rendah", int(counts.get("Rendah", 0)))
            per_course = (result.groupby(["Course_name", "risk_level"]).size().unstack(fill_value=0)
                          .reindex(columns=["Rendah", "Sedang", "Tinggi"], fill_value=0))
            st.markdown("**Sebaran tingkat risiko per program studi**")
            st.bar_chart(per_course, color=["#88BF4D", "#F9D45C", "#EF8C8C"])
            show_cols = (["Status"] if "Status" in result.columns else []) + \
                ["Course_name", "Age_at_enrollment", "Tuition_fees_up_to_date", "Curricular_units_1st_sem_approved",
                 "Curricular_units_2nd_sem_approved", "dropout_probability", "risk_level"]
            st.dataframe(result.sort_values("dropout_probability", ascending=False)[show_cols],
                         use_container_width=True)
            st.download_button("Unduh hasil prediksi", result.to_csv(index=False), "hasil_prediksi_dropout.csv", "text/csv")

with tab_about:
    st.markdown(f"""
**Algoritma:** Random Forest Classifier (`{meta['best_params']}`)

**Target:** 1 = Dropout, 0 = Graduate. Mahasiswa Enrolled tidak dipakai untuk pelatihan karena status akhirnya belum diketahui.

**Kategori risiko:** Tinggi jika probabilitas >= {TH['Tinggi']}, Sedang jika >= {TH['Sedang']}, selain itu Rendah.

**Performa pada data uji:** {", ".join(f"{k} = {v}" for k, v in meta['test_metrics'].items())}

Fitur paling berpengaruh adalah jumlah unit kurikuler yang lulus dan rata-rata nilai di semester 1 dan 2,
status pembayaran UKT, nilai masuk, dan usia saat masuk.
""")
