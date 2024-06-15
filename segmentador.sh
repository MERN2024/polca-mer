
EXTEN=("mp3" "wav" "mp4" )
CARPETA=""

trim () {
    local s2 s="$*"
  until s2="${s#[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  until s2="${s%[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  echo "$s"
}


for e in ${EXTEN[@]}; 
do  
	find *  -name '*.'$e -type f | xargs -0 |while read audio ;do
		
		 ffmpeg -nostdin -i $(trim "$audio") -f segment  -segment_time 15 -c copy  $(dirname $(trim "$audio"))"/"$(basename -s .mp3 $(trim "$audio"))'%03d.'$e
		  
		done	

done



