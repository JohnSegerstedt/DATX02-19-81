import mpyq
import pprint

pp = pprint.PrettyPrinter(indent=4)

print("hello")

archive = mpyq.MPQArchive("replays/3.16.1-Pack_1-fix/Replays/00a0a1b139395dbbba8058a0ec42128b2356cf92abd1dd0ae7059692c37124be.SC2Replay")
contents = archive.header['user_data_header']['content']
print(archive.files)
from s2protocol import versions
header = versions.latest().decode_replay_header(contents)
baseBuild = header['m_version']['m_baseBuild']
protocol = versions.build(baseBuild)
contents = archive.read_file('replay.game.events')



