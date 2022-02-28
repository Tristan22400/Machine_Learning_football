import {Component, OnDestroy, OnInit} from '@angular/core';
import {Subscription} from 'rxjs';
import { EndSeason } from './endseason.model';
import { EndSeasonApiService } from './endseason-api.service';

@Component({
  selector: 'teams',
  templateUrl: './endseason.component.html',
  styleUrls: ['./endseason.component.css']
})

export class EndSeasonComponent implements OnInit
{
  endSeasonListSub: Subscription;
  endSeasonList: EndSeason[];
  displayedColumns: string[] = ['Rank', 'Team', 'Points', 'Wins','Draws','Losses','Goals_scored','Goals_conceded','See_more'];


  constructor(private endseasonApi: EndSeasonApiService)
  {
  }

  

  ngOnInit()
  {
  

    this.endSeasonListSub = this.endseasonApi
          .getEndSeason()
          .subscribe(result => 
            {this.endSeasonList = result;}, console.error);
    const self = this;
  }

}
