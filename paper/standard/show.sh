tail -n 1 wts/*/*.log | grep -v ^$ | tr -d '\n' | tr '>' '\n' | grep wts | sed s/==//g | tr -d '<' | sed s/.*wts\\\///g | sed s/\\\/train.log//g | cut -f 1,4- -d ' ' | sed 's/ \[/  \t[/g'
