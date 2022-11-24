for directory in hop initial; do
    echo
    echo "Entre "$directory
    cd $directory/build
    cmake --build .
    cd ../..
done
