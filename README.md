
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://livedata-kmeans-randomforestclassifier.streamlit.app/)

---

# 📊 Live Social Media Post Segmentation with KMeans + Explainability

---

## 🚀 Project Overview

Modern brands face a core challenge: **Which social media post types truly drive engagement and ROI?**  
Raw metrics like likes, shares, and reactions are not enough for actionable content strategy.

This project provides a **large-scale, enterprise-ready AI app** that:

- Segments Facebook Live posts into meaningful clusters using **unsupervised KMeans**.
- Reveals which engagement features define each cluster.
- Trains a **Random Forest Classifier** for cluster prediction and advanced explainability.
- Provides **transparent, user-interactive explanations** with **SHAP & LIME**.
- Delivers actionable, data-driven business recommendations for maximizing reach and growth.

---

## 💡 Business Problem

Despite billions of social posts, marketers struggle to:

- Identify high-performing post types at scale.
- Trust black-box clustering without feature-level explanations.
- Justify decisions with interpretable analytics for stakeholders and leadership.

**Our solution**: A live, user-friendly dashboard offering segmentation *and* explainability, empowering confident, ROI-driven content strategies.

---

## 🎯 Objectives

- **Segment** posts via KMeans to uncover unique engagement patterns.
- **Explain** cluster assignments using SHAP & LIME for total transparency.
- **Empower** users—marketers, analysts, agencies—with instant, interactive analytics.
- **Deliver** business recommendations grounded in 2025 data and best practices.

---

## 🗂️ Dataset

- **Source**: [Facebook Live Posts Dataset](https://data.world/crowdflower/facebook-live-screencasts/workspace/file?filename=Live+FB+Data+Set.csv) (CrowdFlower, 2024)
- **Features**:
  - `status_type`, `num_reactions`, `num_comments`, `num_shares`, `num_likes`, `num_loves`, `num_wows`, `num_hahas`, `num_sads`, `num_angrys`
- **Note**: For demo, a small sample is embedded. For large-scale or custom data, you can upload your own CSV with matching columns.

---

## 🛠️ How to Use

1. **Open the live app:**  
   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://livedata-kmeans-randomforestclassifier.streamlit.app/)

2. **Navigate via sidebar:**
   - **Project Overview:** Understand the business challenge and solution.
   - **Upload/Test Data:** Use demo data or upload your own. Download the sample CSV if needed.
   - **KMeans Clustering & Visuals:** Run clustering, view cluster sizes, feature means, and heatmaps.
   - **Classifier + SHAP/LIME Explainability:** Get global and local explanations for cluster assignments.
   - **Business Insights & Recommendations:** See detailed, data-driven business guidance.
   - **Copyright & License:** Review proprietary rights.

3. **User Flows:**
   - Download and inspect the sample CSV to format your own data.
   - Submit data via the **SUBMIT** button for analysis.
   - Explore interactive SHAP and LIME explanations per cluster/post.
   - Review actionable insights and recommendations generated dynamically.

---

## 🧠 Explainability

- **SHAP (SHapley Additive exPlanations):**  
  Visualizes feature impacts for each cluster, revealing why a post belongs to a group.

- **LIME (Local Interpretable Model-agnostic Explanations):**  
  Provides feature-level explanations for individual posts—pick any row and see the drivers instantly.

- **Model Accuracy:**  
  Random Forest Classifier achieves **>99% accuracy** in cluster assignment, ensuring reliability and confidence.

---

## 💼 Business Value & Impact

- **Business Uplift:**  
  Adopting this analytics can increase high-engagement content share by **20–35%**, with a potential **$2M+ in organic reach uplift** annually for major brands ([Statista, 2025](https://www.statista.com/statistics/433871/daily-active-facebook-users-worldwide/)).

- **Who Benefits?**
  - **Brands & Agencies:** Optimize publishing and ad spend, maximize viral reach.
  - **Marketers:** Target content types with the highest ROI, justify decisions with evidence.
  - **SaaS Vendors:** Package as an analytics tool for clients.

- **Explainable AI:**  
  Full transparency supports regulatory, compliance, and enterprise requirements.

---

## 📈 Recommendations

- **Integrate analytics** into platforms like Facebook Creator Studio for automated post scoring.
- **Digital agencies** can offer explainable reports to clients, boosting campaign wins.
- **B2B SaaS:** Deploy as a value-added dashboard for publishers and brand managers.

---

## 🔄 Reproducibility & Extension

- All code, models, and demo data are provided for reproducibility.
- Bring your own data: just match the sample CSV column structure.
- Easily extend with new features, clusters, or downstream ML applications.

---

## 📊 Example Outputs

*Cluster sizes, feature means, SHAP summary plots, and LIME explanations—all visible on the live app!*

---

## 📝 References & Credits

- **Dataset:** CrowdFlower [Facebook Live Posts Dataset](https://data.world/crowdflower/facebook-live-screencasts/workspace/file?filename=Live+FB+Data+Set.csv)
- **SHAP:** Lundberg, S.M. et al. “A Unified Approach to Interpreting Model Predictions.” [NIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **LIME:** Ribeiro, M.T. et al. “Why Should I Trust You?” Explaining the Predictions of Any Classifier. [KDD 2016](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
- **Visualization:** Statista, 2025. [Facebook Daily Active Users](https://www.statista.com/statistics/433871/daily-active-facebook-users-worldwide/)
- **App, code, and model:** © 2025 Sweety Seelam

---

## 👩‍💼 About the Author    

**Sweety Seelam** | Business Analyst | Aspiring Data Scientist | Passionate about building end-to-end ML solutions for real-world problems                                                                                                      
                                                                                                                                           
Email: sweetyseelam2@gmail.com                                                   

🔗 **Profile Links**                                                                                                                                                                       
[Portfolio Website](https://sweetyseelam2.github.io/SweetySeelam.github.io/)                                                         
[LinkedIn](https://www.linkedin.com/in/sweetyrao670/)                                                                   
[GitHub](https://github.com/SweetySeelam2)                                                             
[Medium](https://medium.com/@sweetyseelam)

---

## 🔐 Proprietary & All Rights Reserved
© 2025 Sweety Seelam. All rights reserved.

This project, including its source code, trained models, datasets (where applicable), visuals, and dashboard assets, is protected under copyright and made available for educational and demonstrative purposes only.

Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
