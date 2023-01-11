//Jérome Wandhammer 2022
//Modifié Gilles 2023

// Permet de segmenter les images issues du nanozoomer

//Paramètres to do




// Etape 1 :
// Dans imageJ, ouvrir le fichier .ndpis
// Sélectionner une série basse résolution (ex. série 5 = 1/16 de la résolution max)
// Utiliser l'outil MultiPoint pour localiser le centre de chaque sections
// "Analyze/Measure" puis "save as" dans un fichier csv
// Rq : les coordonnées sont en microns


// Etape 2 : Macro


// Ouvre le fichier csv et en extrait les coordonnées des points
Table.open("C:/Users/Gilles.DELBECQ/Desktop/Results.csv");
X = Table.getColumn("X");
Y = Table.getColumn("Y");
sz = Table.size;


// Boucler sur le nombre de lignes
// Faire un Crop On Import sur l'image pleine résolution (série 1)
// Pour chaque point :
//  Placer un rectangle de dimension width_1 par height_1
//  Avec le coin supérieur gauche de coordonées x_coordinate_1 ; y_coordinate_1 (calculer à partir du centre de la coupe renseigné par le csv)

// Les points x et y du CropOnImport doit être donné en pixels -> faire conversion
// Taille finale de l'image (width_1*height_1) est en pixel


for (i=0; i<sz; i++) {
	run("Bio-Formats Importer",
	"open=[//equipe2-nas1/Master.EQUIPE2/jerome.WANDHAMMER/Nanozoomer/220504_68_lames1234/68_brain1_x20 - 2022-05-04 14.49.33.ndpis] color_mode=Default crop rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT series_1 x_coordinate_1="+(2.26*X[i]-6500)+"  y_coordinate_1="+(2.26*Y[i]-3000)+" width_1=12500 height_1=8000"); saveAs("Tiff", "D:/Working_Dir/test Brainj/coronal_jerome/lame68NG_test"+i+".tif");
	close();
}

// Les images segmentées sont sauvées avec un nom incrémentiel dans le dossier de notre choix