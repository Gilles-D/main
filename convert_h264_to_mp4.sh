#Convert all h264 files from the current directory into mp4 using MP4Box (Gpac software)

for i in *.h264;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  MP4Box -add "${name}.h264" "${name}.mp4"
done