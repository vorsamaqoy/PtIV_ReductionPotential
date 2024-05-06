# Machine Learning-Based Prediction of Reduction Potentials for Pt<sup>IV</sup> Complexes

<p align="center">
  <img src="https://github.com/vorsamaqoy/PtIV_ReductionPotential/blob/main/images_medium_ci4c00315_0014.gif?raw=true" alt="Testo alternativo">
</p>

## Abstract
Some of the well-known drawbacks of clinically approved Pt<sup>II</sup> complexes can be overcome using six-coordinate Pt<sup>IV</sup> complexes as inert prodrugs, which release the corresponding four-coordinate active Pt<sup>II</sup> species upon reduction by cellular reducing agents. Therefore, the key factor of Pt<sup>IV</sup> prodrug mechanism of action is their tendency to be reduced which, when the involved mechanism is of outer-sphere type, is measured by the value of the reduction potential. Machine learning (ML) models can be used to effectively capture intricate relationships within Pt<sup>IV</sup> complex data, leading to highly accurate predictions of reduction potentials and other properties, and offering significant insights into their electrochemical behavior and potential applications. In this study, a machine learning-based approach for predicting the reduction potentials of Pt<sup>IV</sup> complexes based on relevant molecular descriptors is presented. Leveraging a data set of experimentally determined reduction potentials and a diverse range of molecular descriptors, the proposed model demonstrates remarkable predictive accuracy (MSE = 0.016 V2, RMSE = 0.13 V, R2 = 0.92). Ab initio calculations and a set of different machine learning algorithms and feature engineering techniques have been employed to systematically explore the relationship between molecular structure and similarity and reduction potential. Specifically, it has been investigated whether the reduction potential of these compounds can be described by combining ML models across different combinations of constitutional, topological, and electronic molecular descriptors. Our results not only provide insights into the crucial factors influencing reduction potentials but also offer a rapid and effective tool for the rational design of Pt<sup>IV</sup> complexes with tailored electrochemical properties for pharmaceutical applications. This approach has the potential to significantly expedite the development and screening of novel Pt<sup>IV</sup> prodrug candidates. The analysis of principal components and key features extracted from the model highlights the significance of structural descriptors of the 2D Atom Pairs type and the lowest unoccupied molecular orbital energy. Specifically, with just 20 appropriately selected descriptors, a notable separation of complexes based on their reduction potential value is achieved.

## Introduction
The development of platinum-based chemotherapeutic agents, spearheaded by the accidental discovery of cisplatin's cytotoxicity by Barnett Rosenberg, has led to the approval of several derivatives like carboplatin and oxaliplatin by the FDA since 1978. However, their clinical efficacy is hindered by adverse effects such as tumor resistance and toxicity to healthy tissues. Consequently, Pt<sup>IV</sup> prodrugs have emerged as a potential solution, offering increased stability and controlled activation within the body. These prodrugs, derived from Pt<sup>II</sup> species, are activated by reduction to their Pt<sup>II</sup> counterparts under physiological conditions, facilitating DNA platination and subsequent cytotoxic effects. The reduction process, catalyzed by endogenous reductants, is pivotal in their mechanism of action. Inner- and outer-sphere reduction mechanisms, characterized by direct interaction or lack thereof between the reducing agent and the complex, influence the ease of reduction. Despite numerous studies, establishing a direct correlation between reduction potentials and cytotoxicity remains challenging. Nonetheless, predicting reduction potentials is crucial for understanding Pt<sup>IV</sup> complex reactivity and guiding rational drug design. Quantum-mechanical calculations, particularly density functional theory (DFT), have traditionally been employed for this purpose, but their accuracy depends on computational parameters and models. To address this, Machine Learning (ML) techniques offer a promising alternative. ML models, trained on experimental data and molecular descriptors, can efficiently predict reduction potentials, facilitating the design of Pt<sup>IV</sup> complexes with tailored properties. This study explores the application of ML in predicting reduction potentials, leveraging its ability to identify intricate structure–property relationships. By systematically analyzing a dataset comprising experimental reduction potentials and molecular descriptors, the study aims to uncover the underlying factors influencing reduction potential and optimize ML models for accurate predictions. The proposed ML approach holds significant potential in advancing the field of platinum-based drug design, particularly in drug delivery systems and targeted therapies, offering a more sustainable and cost-effective approach to drug development.

## Conclusion
The study proposed here explores the critical aspects of the understanding of the electrochemical behavior of Pt<sup>IV</sup> complexes by evaluating the effect of different molecular descriptors on the prediction of the reduction potential, which has implications for various applications, such as in medicinal chemistry and pharmaceutics for the identification of efficacious strategies for fighting cancer.
Computational and experimental investigations highlighted that the electrochemical properties and, specifically, the propensity of Pt<sup>IV</sup> complexes to undergo two-electron reduction, measured by the reduction potential in outer-sphere reduction reactions, are determined by the nature of the axial and, although to a lesser extent, equatorial ligands. By predicting the reduction potential, researchers can establish valuable structure–property relationships, facilitating the rational design of new complexes with tailored electrochemical characteristics. While conventional methods such as quantum-mechanical calculations, particularly DFT, have historically been the standard for predicting redox potentials, this study explores an innovative alternative strategy based on ML techniques, which can supply accurate predictions more efficiently. The ML models have been developed based on a data set comprising experimentally determined reduction potentials for diverse Pt<sup>IV</sup> complexes and an extensive array of molecular descriptors, which allowed us to systematically explore the connection between molecular structure, similarity, and reduction potential. Furthermore, the use of feature engineering and model selection processes allowed us to refine the ML approach, with the ETR model emerging as the optimal model. Hyperparameter optimization further improved the ETR performance, enhancing its accuracy and applicability.
The interpretation of the selected features also provided valuable insights into the factors influencing the reduction potential of Pt<sup>IV</sup> complexes. Descriptors related to specific molecular substructures, such as the presence of chlorido and oxalate-like chelating ligands, have been found to correlate with changes in reduction potential. Additionally, the energy of the LUMO orbital reflected an inverse relationship with reduction potential, aligning with Koopmans’ theorem.
In conclusion, this study not only sheds light on the intricate relationship between molecular descriptors and the reduction potential of Pt<sup>IV</sup> complexes, but also presents a rapid and efficient approach for the rational design of Pt<sup>IV</sup> complexes with customized electrochemical properties, opening new avenues for discoveries in chemistry and pharmaceutics.

## Author Information
**V. Vigna** - PROMOCS Laboratory, Department of Chemistry and Chemical Technologies, University of Calabria, Arcavacata di Rende 87036,Italy;  Orcid https://orcid.org/0009-0007-9599-2813; Email: vincenzo.vigna@unical.it  
**T. F. G. G. Cova** - Coimbra Chemistry Centre, Department of Chemistry, Institute of Molecular Sciences (IMS), Faculty of Sciences and Technology, University of Coimbra, Coimbra 3004-535,Portugal;  Orcidh ttps://orcid.org/0000-0002-2840-6091; Email: tfirmino@qui.uc.pt  

**S. C. C. Nunes** - Coimbra Chemistry Centre, Department of Chemistry, Institute of Molecular Sciences (IMS), Faculty of Sciences and Technology, University of Coimbra, Coimbra 3004-535,Portugal;  Orcid https://orcid.org/0000-0002-3060-5719  
**A. A. C. C. Pais** - Coimbra Chemistry Centre, Department of Chemistry, Institute of Molecular Sciences (IMS), Faculty of Sciences and Technology, University of Coimbra, Coimbra 3004-535,Portugal;  Orcid https://orcid.org/0000-0002-6725-6460  
**E. Sicilia** - PROMOCS Laboratory, Department of Chemistry and Chemical Technologies, University of Calabria, Arcavacata di Rende 87036,Italy;  Orcid https://orcid.org/0000-0001-5952-9927  
