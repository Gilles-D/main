// Chemin d'accès du dossier contenant les images
path = "D:/Margot_dataset/";

// Obtenir la liste des noms de fichiers dans le dossier
list = getFileList(path);

// Boucle à travers chaque fichier dans la liste
for (i = 0; i < list.length; i++) {
    // Vérifier que le fichier est une image
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".jpeg") || endsWith(list[i], ".bmp") || endsWith(list[i], ".png")) {
        // Ouvrir l'image
        open(path + list[i]);
        
        // Séparer les canaux
        run("Split Channels");
        
        // Récupérer le nom de fichier sans l'extension
        name = list[i];
        
        // Enregistrer les canaux séparément en tant que PNG avec un suffixe
        selectWindow("C2-" + name);
        
        resetMinAndMax();
        run("Enhance Contrast...", "saturated=0.35 equalize");
        
        
        name_save = substring(list[i], 0, indexOf(list[i], "."));
		path_save="D:/Margot_dataset/tdtomato/";
        saveAs("JPG", path_save + name_save + ".jpg");
        
        // Fermer toutes les fenêtres d'image ouvertes
        close("*");
    }
}
