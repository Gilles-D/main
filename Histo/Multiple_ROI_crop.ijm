// === Duplicate each ROI from ROI Manager and save as "nomimage_sliceXXX.tif" ===

// Récupère l'image originale
origTitle = getTitle();
origDir = getDirectory("image");

// Vérifie qu'il y a des ROIs
n = roiManager("count");
if (n == 0) exit("⚠️ Aucun ROI trouvé dans le ROI Manager !");

// Base de nom = titre complet (avec #1 si présent)
baseName = origTitle;

// Boucle sur tous les ROIs
for (i = 0; i < n; i++) {
    // Re-sélectionne l'image originale à chaque itération
    selectWindow(origTitle);

    // Sélectionne le ROI i
    roiManager("Select", i);

    // Formate l’index : slice001, slice002, etc.
    idx = i + 1;
    if (idx < 10) idxStr = "00" + idx;
    else if (idx < 100) idxStr = "0" + idx;
    else idxStr = "" + idx;

    // Nom de la duplication
    dupTitle = baseName + "_slice" + idxStr;

    // Duplique le ROI (entre crochets pour garder le titre exact)
    run("Duplicate...", "title=[" + dupTitle + "] duplicate all");

    // Sauvegarde le crop
    savePath = origDir + dupTitle + ".tif";
    saveAs("Tiff", savePath);
}

// 🔹 Fermer toutes les images sauf l’originale
list = getList("image.titles");
for (i = 0; i < list.length; i++) {
    if (list[i] != origTitle) {
        selectWindow(list[i]);
        close();
    }
}

// Re-sélectionne l’image originale à la fin
selectWindow(origTitle);

print("✅ " + n + " ROI(s) dupliqués, sauvegardés et toutes les duplications ont été fermées.");
