// === Plot Profile automatique sur plusieurs images ouvertes ===

// Nom du fichier de sortie (CSV)
saveDir = getDirectory("image"); // ou dossier de ton choix
saveFile = saveDir + "Profiles_bilateral_lame1_right_tdTomato.csv";

// Crée le fichier et écrit l'entête
File.saveString("Distance,Image,Intensity\n", saveFile);

// Récupère la liste de toutes les images ouvertes
imgTitles = getList("image.titles");

// Boucle sur toutes les images
for (i = 0; i < imgTitles.length; i++) {
    title = imgTitles[i];
    
    selectWindow(title);
	profile = getProfile();
	
    for (j = 0; j < profile.length; j++) {
        File.append(j + "," + title + "," + profile[j] + "\n", saveFile);
    }

    print("Profil enregistré pour : " + title);
	
}

print("✅ Tous les profils enregistrés dans : " + saveFile);
