// Chemin d'accès du dossier contenant les images
path = "D:/Margot_dataset/tdtomato/jpg/";

// Obtenir la liste des noms de fichiers dans le dossier
list = getFileList(path);

// Boucle à travers chaque fichier dans la liste
for (i = 0; i < list.length; i++) {
    // Vérifier que le fichier est une image
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".jpeg") || endsWith(list[i], ".bmp") || endsWith(list[i], ".png")) {
        // Ouvrir l'image
        open(path + list[i]);
        
        // Récupérer le nom de fichier sans l'extension
        name = list[i];

        run("Scale...", "x=0.25 y=0.25 width=2100 height=1408 interpolation=Bilinear average create");
        
        name_save = substring(list[i], 0, indexOf(list[i], "."));
		path_save="D:/Margot_dataset/tdtomato/jpg/downscaled/";
        saveAs("JPG", path_save + name_save + ".jpg");
        
        // Fermer toutes les fenêtres d'image ouvertes
        close("*");
    }
}
