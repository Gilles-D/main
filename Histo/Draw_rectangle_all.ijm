for (i=0;i<nImages;i++) {
      selectImage(i+1);
      Stack.setChannel(2);
      run("Enhance Contrast", "saturation=35");

      makeRectangle(2500, 936, 1500, 660);

} 

print("Done")