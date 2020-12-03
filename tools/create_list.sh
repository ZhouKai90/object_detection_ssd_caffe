#!/bin/bash
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir=$bash_dir/..
dataset_name="pedestrians"
data_dir=$root_dir/data/$dataset_name/voc
list_dir=$root_dir/data/$dataset_name
sub_dir=ImageSets/Main

echo $bash_dir
for dataset in train val
do
  dst_file=$root_dir/data/$dataset_name/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007
  do
    echo "Create list for $name $dataset..."
    dataset_file=$data_dir/$name/$sub_dir/$dataset.txt
    echo $dataset_file

    img_file=$list_dir/$dataset"_img.txt"
    echo $img_file
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$list_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset=="val" ]
  then
    $root_dir/caffe_ssd/build/tools/get_image_size $data_dir $dst_file $root_dir/data/$dataset_name/$dataset"_name_size.txt"
  fi

  # Shuffle train file.
  if [ $dataset=="train" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
