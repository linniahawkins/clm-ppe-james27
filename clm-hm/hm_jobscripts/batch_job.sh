
for i in {0002..0100}; do
    
    sed 's/num/'$i'/g' template.sh > 'hmatch_'$i'.job'
    qsub 'hmatch_'$i'.job'

done
