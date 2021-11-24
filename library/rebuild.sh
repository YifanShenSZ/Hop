for directory in hop initial; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm lib*
    cmake --build .
    cd ../..
done
