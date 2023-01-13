dir = getDirectory("Choose a Directory");

for (i=0;i<nImages;i++) {

	title = getTitle;
	print(title);
    selectImage(i+1);
    Stack.setChannel(2);

	run("Clear Results");
	profile = getProfile();
	for (j=0; j<profile.length; j++)
	  setResult("Value", j, profile[j]);
	updateResults();
	path = dir+title+".csv";
	saveAs("Results", path);
	} 

print("Done");


    //run("Plot Profile");
    
  //Plot.getValues(x, y);
  //Plot.create("Plot Values", "X", "Y", x, y);
  //Plot.getValues(x, y);
  //for (i=0; i<x.length; i++)
  //    print(x[i], y[i]);