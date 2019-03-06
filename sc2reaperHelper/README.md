

<h2>Version Mismatch</h2>
When using sc2reaper to parse replays, one must have pre-set the replays' versions to avoid version mismatch errors.
To do this, do the following steps:

<h3>Step1</h3>
Make sure you have the 4.X.Y "binary" installed of the replay version you want to parse. This is done by manually double-clicking on a .SC2Replay file with the desired game version. If the replay loads successfully, you have the correct version installed. Otherwise, a window will appear with "DOWNLOADING" as the title which will download the correct game version and then it will load the replay succesfully, which means you now also have the correct version installed.

<h3>Step2</h3>
Go to the following website: https://github.com/Blizzard/s2client-proto/blob/master/buildinfo/versions.json. There, use a "search" function (like ctrl+f) to find the game version you want to parse a replay from. Keep this information at the ready since it will be used in the next step. Ex: If you want to parse a replay from 4.2.1, find the following text:<br>
<i>"base-version": 62848,<br>
"data-hash": "29BBAC5AFF364B6101B661DB468E3A37", <br>
"fixed-hash": "ABAF9318FE79E84485BEC5D79C31262C", <br>
"label": "4.2.1", <br>
"replay-hash": "A7ACEC5759ADB459A5CEC30A575830EC", <br>
"version": 62848<br>
</i>

<h3>Step3</h3>
Go to the following folder: C:\Program Files (x86)\StarCraft II\Versions\. There you should see multiple "BaseXXXXX" folders. Each folder which name does not match with the "base-version" found above, move these elsewhere. Ex: If you want to parse a replay from 4.2.1, then move the other folders so that "Base62848" is the only "BaseXXXXX" folder in this directory.

<h3>Step4</h3>
Find your the following file "... \pysc2\lib\sc2_process.py". I found mine at "C:\Program Files (x86)\Python37-32\Lib\site-packages\pysc2\lib\sc2_process.py". Open this file with admin privilege (= open an IDE like the Python IDLE with admin privileges and then use that IDLE instance to open the file). Add ' "-dataversion", "XXXXXXXXXXXXXXXXXXXXXXXXXX" ' to the 'args' declaration, where the "XXX....XXX" are from "data-hash" found earlier in Step 2. Ex; if you want to parse version 4.2.1, this string is "29BBAC5AFF364B6101B661DB468E3A37" and the 'args' declatartion should look like this:<br>
<i>
args = [<br>
exec_path,<br>
"-listen", self._host,<br>
"-port", str(self._port),<br>
"-dataDir", os.path.join(run_config.data_dir, ""),<br>
"-tempDir", os.path.join(self._tmp_dir, ""),<br>
"-dataversion", "29BBAC5AFF364B6101B661DB468E3A37",<br>
]<br>
</i>

<h3>Step5</h3>
Run the reapHelper.py script as usual. Note - if you ever get a very bland "Cannot open replay" error message, redo Step3 as StarCraft II likes to redownload the newest "BaseXXXXX" folder, ruining everything.

------------------------------------------------------

Parsing with sc2reaper
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
