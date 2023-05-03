// Chemin d'accès du dossier contenant les images
path = "F:/Data/Microscopie/SOD/1231/jpeg/tdtomato/object prediction/todo/";

// Obtenir la liste des noms de fichiers dans le dossier
list = getFileList(path);

// Boucle à travers chaque fichier dans la liste
for (i = 0; i < list.length; i++) {
    // Vérifier que le fichier est une image
    if (endsWith(list[i], ".tif") || endsWith(list[i], ".jpg") || endsWith(list[i], ".jpeg") || endsWith(list[i], ".bmp") || endsWith(list[i], ".png")) {
        // Ouvrir l'image
        open(path + list[i]);
        run("glasbey");
        
        wait(500);
        saveAs("png", path + list[i]);}
}