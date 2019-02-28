Follow instructions @
https://github.com/miguelgondu/sc2reaper

For single file run from cmd like:
sc2reaper ingest C:\Users\da_fo\Desktop\kandidat\sc2reader\ggtracker_231557.SC2Replay

For multiple files in a folder use reapHelper.py from cmd: 
python reapHelper.py <directory>


if you get failed to open map archive:
	make sure you have the appropriate map installed. map packs can be found here:
		https://github.com/Blizzard/s2client-proto#downloads
if you still get failed to open map archive: 	
	https://github.com/Blizzard/s2client-proto/issues/12

If you get faulty version this is because the replay you are attempting to run is not the same version as your installed starcraft II. 
	If you are running linux you can find old starcraft II versions here:
	https://github.com/Blizzard/s2client-proto#downloads