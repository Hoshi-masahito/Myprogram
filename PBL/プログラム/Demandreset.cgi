#!/usr/bin/perl
$title="4Dクリーナー";

open MYFILE, ">Demandcount.txt";
print MYFILE "0";
close MYFILE;

open MYFILE, ">gomibako.txt";
print MYFILE "ゴミ箱はあふれていません";
close MYFILE;

use Net::SMTP;
use Encode qw(from_to encode);

sub send {
    $from = $_[0];
    $mailto = $_[1];
    $subject = 'test';

    #メールのヘッダ
    $header = << "MAILHEADER";
From: $from
To: $mailto
Subject: $subject
Mime-Version: 1.0
Content-Type: text/plain; charset = "ISO-2022-JP"
Content-Transfer-Encoding: 7bit
MAILHEADER

    #メール本文
    $message = << "__BODY__" ;
$_[2]
__BODY__

    #文字コードをJISに変換
    from_to($message, 'utf-8', 'iso-2022-jp');

    #$smtp = Net::SMTP->new('mail.cc.ibaraki-ct.ac.jp');
    $smtp = Net::SMTP->new('172.16.12.19');
    $smtp->mail($from);
    $smtp->to($mailto);
    $smtp->data();
    $smtp->datasend("$header\n");
    $smtp->datasend("$message\n");
    $smtp->dataend();
    $smtp->quit;
}
&send('st14d19@gm.ibaraki-ct.ac.jp','st14d36@gm.ibaraki-ct.ac.jp',"クラスの掃除が終了しました!");


print << "EOT";
Content-Type: text/html

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xml:lang="ja" lang="ja">

<head>

<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta http-equiv="Content-Script-Type" content="text/javascript" />
<meta http-equiv="Content-Style-Type" content="text/css" />
<meta name="viewport" content=width-device-width, user-scalable=no, maximum-scale=1" />

<title>4Dクリーナー</title>

<style type="text/css">
.contents{
	width: 752px;
	border: 1px solid #FF9900;
	margin: 5px auto;
}
</style>
</head>
<body>
<div class= "contents">
<p>お掃除お疲れ様でした。<br></p>
<a href="4D.cgi">トップページに戻る</a>
</div>
</body>
</html>

EOT
