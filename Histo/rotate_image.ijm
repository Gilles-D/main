// "Rotate Image"
// This macro rotates an image based on the angle
// of a straight line selection. The image is rotated
// so objects oriented similarly to the line 
// selection become horizontal.

  macro "Rotate Image" {
      requires("1.33o");
      getLine(x1, y1, x2, y2, width);
      if (x1==-1)
           exit("This macro requires a straight line selection");
      angle = (180.0/PI)*atan2(y1-y2, x2-x1);
     Stack.setChannel(1);
     run("Arbitrarily...", "angle="+angle+" interpolate");
     Stack.setChannel(2);
     run("Arbitrarily...", "angle="+angle+" interpolate");
     //Stack.setChannel(3);
     //run("Arbitrarily...", "angle="+angle+" interpolate");

  }




