import {Component, OnDestroy, OnInit} from '@angular/core';
import {Subscription} from 'rxjs';
import {Match} from './match.model';
import {MatchsApiService} from './matchs-api.service';
import { DatePipe } from '@angular/common';

@Component({
  selector: 'exams',
  templateUrl: './matchs.component.html',
  styleUrls: ['./matchs.component.css']

})

export class MatchComponent implements OnInit
{
    matchListSub: Subscription;
    matchList: Match[];
    todayDate : string;

    constructor(private matchsApi: MatchsApiService,    private datepipe : DatePipe,)
    {
      this.todayDate =this.datepipe.transform((new Date), 'yyyy-MM-dd');
    }

    ngOnInit()
    {
      this.matchListSub = this.matchsApi
            .getMachtes(this.todayDate)
            .subscribe(result => 
              {this.matchList = result;}, console.error);
      const self = this;
    }

    




}
