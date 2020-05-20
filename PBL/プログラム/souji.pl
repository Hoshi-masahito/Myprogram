open FILE, "soujitoban.txt";
@toban = <FILE>;
open FILE1, "count.txt";
$count = <FILE1>;
$a = scalar @toban;
$b = scalar $count;
for($i = 0;$i < $a;$i++){
	if($b % 6 == 0 || $b % 7 == 0) {
		if($i < 7) {
			push @soji, "$toban[$i]";
			$toban[$i] =~ s/(  | )+//g;
		} else {
			push @nosoji, "$toban[$i]";
			$toban[$i] =~ s/(  | )+//g;
		}
	} else {
		if($i < 6) {
			push @soji, "$toban[$i]";
			$toban[$i] =~ s/(  | )+//g;
		} else {
			push @nosoji, "$toban[$i]";
			$toban[$i] =~ s/(  | )+//g;
		}
	}
}
$count++;
close FILE1;
print @soji;
close FILE;

open (FILE1,">","count.txt");
	print FILE1 $count;
close (FILE1);
open (FILE,">","soujitoban.txt");
	print FILE @nosoji;
	print FILE @soji;
close (FILE);