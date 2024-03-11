# Project3
Co-author recommender system
1. In order to use the system, make sure to have the following versions of:
- Flask==1.1.2
- Flask-Cors==3.0.9
2. Create a folder named "Data_Project3" in the directory containing the cloned folder, 
and place file db.sqlite3 there. Also note that sub-databases will be created and stored in
 this folder
3. Run file 'query_code.py' in folder BE through terminal or GUI and open file 
'bootstrap_template.html' in folder FE on a web browser (Google Chrome)
4. When finishing the last step - calculate scores, result files will be created and 
stored in folder Result with the name given or with 
the format: Data_JournalID_FromDate_ToDate_WeightType_LabelType.csv in which:
	>- JournalID: list of ID(s) of paper journal chosen 
(available IDs: 21, 22, 23, 24)
	>- FromDate: year chosen to start selecting papers 
	>- ToDate: year chosen to end selecting papers
	>- WeightType: weighted or unweighted (considered when calculating scores)
	>- LabelType: static or dynamic (2 labeling schemes)
>For example: Data_1028_21_22_2016_2017_unweighted_dynamic.csv
5. After having dataset, one could start using training phase and then recommnendation 
phase:
- Open file "traing_recommend.html" contained in folder FE on a web browser 
(Google Chrome)
- Run file 'train_recommend.py' contained in folder BE through terminal or GUI
- Training phase: 
	+ Make sure to have sklearn and pandas installed	
	+ Choose file dataset, choose train-test-split percentage. After training,
 model and scaler will be saved at folder Models, and training results will be displayed.
- Recommendation pahse:
	+ Choose Topic of journal, year begin, year end to generate the co-author graph,
 then choose author to recommend and model. Top 5 candidates found by network links 
and affiliation will be displayed.
6. For further source code details or any requirement, please contact 
lam.tt173226@sis.hust.edu.vn
