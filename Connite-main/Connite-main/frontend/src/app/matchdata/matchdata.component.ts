import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Subscription } from 'rxjs';
import { DataApiService } from './matchdata-api.service';
import { Calcul } from './calcul.model';

@Component({
  selector: 'app-matchdata',
  templateUrl: './matchdata.component.html',
  styleUrls: ['./matchdata.component.css']
})
export class MatchdataComponent implements OnInit {
  homeTeam: any;
  awayTeam: string;
  odds_homeTeam = 0;
  odds_awayTeam = 0;
  objectiveSubscription : Subscription;
  objective_homeTeam : string;
  objective_awayTeam : string;
  importance_homeTeam = 0;
  importance_awayTeam = 0;
  commentaire_homeTeam = "Ce match joue peu pour les objectifs de cet équipe!";
  commentaire_awayTeam = "Ce match joue beaucoup pour les objectifs de cet équipe!";

  constructor(private route: ActivatedRoute, public DataApi: DataApiService)
   {

    }

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      this.homeTeam = this.route.snapshot.paramMap.get('team1')
      this.awayTeam = this.route.snapshot.paramMap.get('team2')


    this.objectiveSubscription = this.DataApi
    .getObjectives(this.homeTeam)
    .subscribe(result => 
      {this.objective_homeTeam = result[0]["Top"];}, console.error);

    this.objectiveSubscription = this.DataApi
    .getObjectives(this.awayTeam)
    .subscribe(result => 
      {this.objective_awayTeam = result[0]["Top"];}, console.error);


    this.objectiveSubscription = this.DataApi
      .getCalculs(this.homeTeam)
      .subscribe(result => 
        {this.odds_homeTeam = result[0].Odds, this.importance_homeTeam=result[0].Importance;}, console.error);
      
    this.objectiveSubscription = this.DataApi
      .getCalculs(this.awayTeam)
      .subscribe(result => 
          {this.odds_awayTeam = result[0].Odds, this.importance_awayTeam=result[0].Importance;}, console.error);

      
    const self = this; 
    });
  }

}