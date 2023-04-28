// Chemin d'accès du dossier contenant les images
path = "//equipe2-nas1/Gilles.DELBECQ/Data/Microscopie/6567/Tiff/";
path_save="//equipe2-nas1/Gilles.DELBECQ/Data/Microscopie/6567/downscaled/";


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
        wait(1000);
        name_save = substring(list[i], 0, indexOf(list[i], "."));
		
        saveAs("JPG", path_save + name_save + ".jpg");
        
        // Fermer toutes les fenêtres d'image ouvertes
        close("*");
    }
}
