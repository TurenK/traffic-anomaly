# traffic-anomaly
1.data processing: use an API (OSM mapping) from Apache to get traffic flow of each road (main roads inside Beijing Four Ring); remove meaningless data. 
2.traffic mode: use NMF to get traffic mode and compare them to get similiarity. 
3.classification: linearly add traffic mode and physical distance with a factor to represent each road --> use kmeans with batch to cluster. 
4.prediction: use knn to predict traffic condition in real time. 
