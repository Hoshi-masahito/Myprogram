#!/usr/bin/perl

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

    print "Sending mail...\n";
    #$smtp = Net::SMTP->new('mail.cc.ibaraki-ct.ac.jp');
    $smtp = Net::SMTP->new('172.16.12.19');
    $smtp->mail($from);
    $smtp->to($mailto);
    $smtp->data();
    $smtp->datasend("$header\n");
    $smtp->datasend("$message\n");
    $smtp->dataend();
    $smtp->quit;
    print "Done.\n";
}

