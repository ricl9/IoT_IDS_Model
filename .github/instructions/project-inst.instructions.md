---
applyTo: '**'
---
In this project, you'll build a sophisticated two-stage intrusion detection system (IDS) for IoT networks that mirrors real-world cybersecurity challenges. Using the CIC IoT-DIAD 2024 dataset, you'll combine the strengths of both anomaly-based and signature-based detection methods to create a robust system that can identify threats while minimizing false alarms.
The core idea is elegant: first, you'll use unsupervised learning to spot anything unusual in network packets (anomaly detection), then apply supervised learning on flow-level data to verify these findings and reduce false positives. This hybrid approach reflects how modern security systems balance broad threat detection with precise identification.

You'll work with the CIC IoT-DIAD 2024 dataset, which captures network traffic from IoT devices under various attack scenarios. This dataset is particularly valuable because it includes both packet-level and flow-level features, allowing you to explore detection at different granularities.
We focus on 5 attack types in the dataset and ignore the rest:
• DoS-TCP_Flood
• DDoS-TCP_Flood
• DNS Spoofing
• Cross-Site Scripting (XSS)
• Brute Force

## Phase 1: Data Preparation
Building Your Dataset Generator
Start by creating a Python function that randomly samples data to simulate realistic network conditions. Your function should generate:
• Approximately 200,000 benign traffic samples (97-98% of your dataset)
• Between 4,000-6,200 attack samples (2-3% of your dataset)
• Attack traffic should be randomly distributed across all five attack types
This imbalanced distribution reflects real-world scenarios where attacks are relatively rare events. Make sure your randomization is properly seeded so you can generate different dataset compositions for testing robustness.
Data Preprocessing
Once you have your dataset, prepare it for modeling by:
• Handling missing values and outliers thoughtfully (document your approach)
• Applying appropriate scaling or normalization for network traffic features
• Considering feature selection or dimensionality reduction if it improves performance
• Addressing any data quality issues specific to network traffic
Remember to justify your preprocessing decisions – they should make sense for network security data.

## Phase 2: Anomaly-Based Detection (Unsupervised Learning)
Building Your Packet-Level Detector
This is where you'll implement your first line of defense. Using only packet-level features and without looking at labels during training, build an unsupervised model that can distinguish normal from potentially malicious traffic.
You have flexibility in your approach here. Consider methods like:
• Autoencoders (which we discussed in class) that learn to reconstruct normal traffic
• K-means clustering to group similar traffic patterns
• A combination approach where autoencoders provide reconstruction errors that K-means then clusters
• Any other unsupervised method you believe would work well
• Show the clusters found in a proper visualization like t-SNE or a PCA based scatter plot with proper boundaries annotated and include it in your report
The key challenge is setting appropriate thresholds for generating alerts. You'll need to balance sensitivity (catching attacks) with specificity (avoiding false alarms).

## Phase 3: Signature-Based Refinement (Supervised Learning)
### Understanding Flow-Level Data
Here's an important detail: each packet in your dataset has corresponding flow information identified by the format [source_IP]-[destination_IP]-[sender_port]-[receiver_port]. Flows lasting longer than 2 minutes are split into multiple segments, so you'll need to aggregate these appropriately when processing.
### Optional Enhancement: Flow-Level Anomaly Detection
Before jumping into supervised learning, consider implementing your anomaly detection approach on flow-level data as well. This optional step can provide valuable insights:
• How does flow-level anomaly detection compare to packet-level?
• Do flow-based features naturally capture certain attack patterns better?
• Can insights from flow-level anomaly detection inform your supervised model with additional useful features?
### Building Your Signature-Based Classifier
Now create a supervised model that examines the packets your anomaly detector flagged as suspicious. Using flow-level features and actual labels, this model acts as a second opinion, reducing false positives while maintaining high detection rates.
For this phase:
1. Generate a new random flow-level dataset using your sampling function (again with the same amount and portion of normal and attack data)
2. For flagged packets from Phase 2, locate their corresponding flow data
3. Remember to aggregate flow segments (those 2-minute splits) appropriately using group by and averaging
4. Train a supervised classifier to identify the proper class out of 6 possible
5. If you completed the optional flow-level anomaly detection, consider using those features to enhance your supervised model
6. As discussed in the class, methods like attention network and GNNs could add more information regarding topology and connections and might help with the performance
7. If you are trying different models and methods, document the results on the validation to show your evolution and progress


## Evaluation Framework
### Performance Metrics
For Packet-Level Detection (Phase 2):
• Overall precision, recall, and F1-score
• Detection rates for each individual attack type
• False Positive Rate (FPR) and False Negative Rate (FNR)
• ROC curves and AUC scores
• Confusion matrices to understand misclassification patterns
• Analysis of how you handled the severe class imbalance
### For Flow-Level Refinement (Phase 3):
• Quantify the reduction in false positives
• Overall system accuracy after both stages
• Performance breakdown by attack class
• Combined precision and recall metrics
• Computational overhead and time complexity comparison
### Comparative Analysis:
Provide insights on:
• How does your two-stage approach compare to using either method alone?
• Which attacks are easiest to detect at the anomaly-based packet level and why?
• How do DDoS and DoS attacks cluster differently in your unsupervised analysis?
• What's the statistical significance of your improvements?


### Technical Report
Structure your report to tell the story of your investigation:
Introduction and Approach Set up the problem and explain your methodology choices. Why did you choose specific algorithms? How does your approach address the challenges of IoT network security?
### Implementation Journey Detail your implementation process, including:
• Data preprocessing decisions and their rationale
• Model architectures and hyperparameter tuning strategies
• Training procedures and optimization approaches
• Any challenges you encountered and how you solved them
### Results and Analysis Present your findings with clear visualizations:
• Performance metrics across both phases
• ROC curves, precision-recall curves, and confusion matrices
• Performance breakdown by attack type (including both DDoS-TCP_Flood and DoS-TCP_Flood variants)
• Statistical analysis demonstrating improvement significance
### Key Insights This is where you demonstrate deep understanding:
• Does flow duration or packet count correlate with classification accuracy? Show the data.
• Which attacks benefit most from your two-stage approach and why?
### Conclusions and Future Directions
Summarize your contributions, acknowledge limitations, and suggest improvements.