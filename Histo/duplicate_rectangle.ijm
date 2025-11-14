// === Macro Fiji : créer un nouveau rectangle (1500x250) accolé à droite de chaque ROI existante ===
// Les nouvelles ROIs sont ajoutées à la fin sans être retraitées dans la boucle.

newWidth = 1500;
newHeight = 250;

// Récupère le nombre initial de ROIs (pour ne pas boucler sur les nouvelles)
n = roiManager("count");
if (n == 0) exit("Aucune ROI trouvée dans le ROI Manager !");

for (i = 0; i < n; i++) {
    roiManager("Select", i);

    // Récupérer les coordonnées du rectangle existant
    getSelectionCoordinates(xPoints, yPoints);
    // Pour un rectangle : 
    // x1,y1 = coin haut gauche
    // x2,y2 = coin haut droit
    // x3,y3 = coin bas droit
    // x4,y4 = coin bas gauche

    x2 = xPoints[1];
    y2 = yPoints[1];
    x4 = xPoints[3];
    y4 = yPoints[3];

    // Nouveau rectangle : accolé au bord droit du précédent
    // => le coin haut gauche du nouveau = (x2, y2)
    // => le coin bas gauche du nouveau = (x4, y4)
    newX = x2;
    newY = y2; // y2 et y4 devraient être alignés verticalement
    // On crée avec newWidth/newHeight (fixes)
    makeRectangle(newX, newY, newWidth, newHeight);

    // Ajouter au ROI Manager (à la fin)
    roiManager("Add");

    print("ROI", i, ": new rectangle ajouté à x=", newX, " y=", newY);
}

print("Done — tous les nouveaux rectangles accolés à droite ont été ajoutés !");
